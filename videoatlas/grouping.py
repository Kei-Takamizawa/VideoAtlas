# このモジュールは検出した顔を慎重に人物へ割り当て、十分な根拠がある人物だけを統合します。
"""Conservative local face grouping for VideoAtlas."""

# 辞書の値を自動的に空のリストにするために読み込みます。
from collections import defaultdict
# 内部の比較情報を分かりやすいデータ型にするために読み込みます。
from dataclasses import dataclass
# 特徴量の長さと距離を計算するために読み込みます。
import math
# 顔画像の保存先を扱うために読み込みます。
from pathlib import Path
# 新しい顔と人物へ一意な識別子を付けるために読み込みます。
from uuid import uuid4
# 型注釈だけで解析結果の型を参照するために読み込みます。
from typing import TYPE_CHECKING

# 保存層と同じ顔・人物・動画の型を使います。
from .storage import FaceRecord, PersonRecord, VideoRecord

# 型検査時だけ解析モジュールを読み込み、解析側からの循環読み込みを避けます。
if TYPE_CHECKING:
    # 解析器が返す顔の形を公開関数の型注釈に使います。
    from .analyzer import FaceSample


# 特徴量が不正な場合に無理な人物照合を行わないための距離です。
NO_MATCH = math.inf


# 二つの特徴量のコサイン距離を計算します。
def _distance(left: list[float] | None, right: list[float] | None) -> float:
    # 片方が欠けるか次元が異なる場合は照合できません。
    if not left or not right or len(left) != len(right):
        # 無限大はどの人物にも一致しないことを表します。
        return NO_MATCH
    # 各成分の積を足して内積を求めます。
    dot = math.fsum(a * b for a, b in zip(left, right))
    # 一つ目の特徴量の長さを求めます。
    left_length = math.sqrt(math.fsum(value * value for value in left))
    # 二つ目の特徴量の長さを求めます。
    right_length = math.sqrt(math.fsum(value * value for value in right))
    # ゼロや非数値の特徴量は比較対象から外します。
    if not math.isfinite(dot) or not math.isfinite(left_length * right_length) or left_length == 0 or right_length == 0:
        # 不正な特徴量から人物を推測しません。
        return NO_MATCH
    # 値が小さいほど顔の特徴が近い距離を返します。
    return max(0.0, 1.0 - dot / (left_length * right_length))


# 二つの顔枠が重なる割合を面積で求めます。
def _overlap(left: tuple[float, float, float, float] | None, right: tuple[float, float, float, float] | None) -> float:
    # 顔枠がない場合は位置による一致を判断できません。
    if left is None or right is None:
        # 重なりなしとして扱います。
        return 0.0
    # 二つの左端のうち右側を共通領域の左端にします。
    x1 = max(left[0], right[0])
    # 二つの上端のうち下側を共通領域の上端にします。
    y1 = max(left[1], right[1])
    # 二つの右端のうち左側を共通領域の右端にします。
    x2 = min(left[0] + left[2], right[0] + right[2])
    # 二つの下端のうち上側を共通領域の下端にします。
    y2 = min(left[1] + left[3], right[1] + right[3])
    # 幅か高さが負なら重なりはゼロです。
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    # 二つの顔枠を合わせた面積を求めます。
    union = left[2] * left[3] + right[2] * right[3] - intersection
    # 壊れた顔枠の面積では人物を判断しません。
    if union <= 0:
        # 比較できない顔枠の重なりをゼロとして返します。
        return 0.0
    # 共通面積を全体面積で割った割合を返します。
    return intersection / union


# ある人物から比較に使う異なる顔向きの代表を最大八件選びます。
def _add_representative(face: FaceRecord, representatives: dict[str, list[FaceRecord]]) -> None:
    # 除外顔、未分類顔、比較不能な顔は代表にできません。
    if face.excluded or face.person_id is None or face.embedding is None:
        # 代表一覧を変えずに終わります。
        return
    # 対象人物の現在の代表一覧を取得します。
    examples = representatives.setdefault(face.person_id, [])
    # 同じ顔を二度登録しません。
    if any(example.id == face.id for example in examples):
        # 既存の代表を維持します。
        return
    # 最初の一枚は基準となる代表にします。
    if not examples:
        # この顔を最初の代表として保存します。
        examples.append(face)
        # ほかの代表との比較は必要ありません。
        return
    # 既存代表から最も近い顔との距離を求めます。
    novelty = min(_distance(face.embedding, example.embedding) for example in examples)
    # ほぼ同じ特徴の画像なら代表を増やしません。
    if novelty < 0.10:
        # 重複した姿勢の顔は保存済み代表で足ります。
        return
    # 八枚未満なら異なる顔を追加します。
    if len(examples) < 8:
        # 新しい向きの顔を代表に加えます。
        examples.append(face)
        # 代表追加後の置き換えは不要です。
        return
    # 最初の基準顔を残し、残りから最も重複した代表を選びます。
    separations = [(min(_distance(examples[index].embedding, other.embedding) for other_index, other in enumerate(examples) if other_index != index), index) for index in range(1, len(examples))]
    # 距離が最も小さい代表は別の代表とよく似ています。
    smallest_separation, redundant_index = min(separations)
    # 新顔がその重複より明確に異なる場合だけ置き換えます。
    if novelty > smallest_separation + 0.03:
        # 基準顔以外の重複代表を新しい向きの顔にします。
        examples[redundant_index] = face


# 短時間で同じ場所に映り続けた人物の候補を計算します。
def _temporal_score(previous: FaceRecord, sample: "FaceSample", maximum_gap: float) -> float:
    # 前の顔からの経過秒数を求めます。
    gap = sample.second - previous.second
    # 同時刻や長い空白をまたぐ顔は追跡しません。
    if gap <= 0 or gap > maximum_gap:
        # 条件外の候補であることを返します。
        return NO_MATCH
    # 顔枠の半分以上が重なることを要求します。
    overlap = _overlap(previous.bounding_box, sample.bounding_box)
    # 位置の根拠が弱い場合は追跡しません。
    if overlap < 0.5:
        # 条件外の候補であることを返します。
        return NO_MATCH
    # 前後の顔特徴量を比較します。
    distance = _distance(previous.embedding, sample.embedding)
    # 特徴が大きく違う顔を位置だけで結びません。
    if distance > 0.45:
        # 条件外の候補であることを返します。
        return NO_MATCH
    # 特徴、位置、時間差を合わせ、低い値を強い候補にします。
    return distance + 0.2 * (1.0 - overlap) + 0.01 * gap / maximum_gap


# 複数候補や同時刻の別顔と競合しない追跡人物だけを選びます。
def _temporal_person(sample: "FaceSample", peers: list["FaceSample"], recent: dict[str, FaceRecord], occupied: set[str], maximum_gap: float) -> str | None:
    # 特徴量がない顔を位置だけで人物に結びません。
    if sample.embedding is None:
        # 判断不能な顔として未分類にします。
        return None
    # 各人物の直近顔から有効な候補を作ります。
    candidates = sorted((_temporal_score(face, sample, maximum_gap), person_id) for person_id, face in recent.items() if person_id not in occupied)
    # 有効な候補が一人もいなければ追跡できません。
    if not candidates or not math.isfinite(candidates[0][0]):
        # 追跡による割当は行いません。
        return None
    # 最もよく一致する候補の値と人物を取ります。
    best_score, best_id = candidates[0]
    # 次点との差が小さい場合は誤分類を避けます。
    if len(candidates) > 1 and candidates[1][0] - best_score < 0.05:
        # どちらの人物か分からないため未分類にします。
        return None
    # 同じフレームの別顔もその人物へ結び付きそうか調べます。
    if any(peer is not sample and _temporal_score(recent[best_id], peer, maximum_gap) <= best_score + 0.05 for peer in peers):
        # 同時出現する別人と一人を混ぜないようにします。
        return None
    # 一意に追跡できた人物のIDを返します。
    return best_id


# 特徴量と短時間追跡に矛盾がない場合だけ人物を割り当てます。
def _cluster_person(sample: "FaceSample", representatives: dict[str, list[FaceRecord]], occupied: set[str], temporal_id: str | None, thumbnail_path: str, new_people: list[PersonRecord]) -> str | None:
    # 特徴量を作れない顔は未分類に残します。
    if sample.embedding is None:
        # 無理な人物推定は行いません。
        return None
    # 各人物の最大八枚の代表顔から最短距離を求めます。
    candidates = sorted((min(_distance(sample.embedding, face.embedding) for face in examples), person_id) for person_id, examples in representatives.items() if person_id not in occupied and examples)
    # 慎重なしきい値に届かなかった人物を候補から除きます。
    candidates = [candidate for candidate in candidates if candidate[0] <= 0.25]
    # 上位二候補の距離差が小さければ人物を決めません。
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.05:
        # 曖昧な顔を未分類として返します。
        return None
    # 代表顔から一人に絞れた場合を扱います。
    if candidates:
        # 最も近い候補の人物IDを取得します。
        best_id = candidates[0][1]
        # 時間追跡が異なる人物を示す場合は両方の根拠を採用しません。
        if temporal_id is not None and temporal_id != best_id:
            # 相反する根拠による誤分類を防ぎます。
            return None
        # 矛盾がなければ既存人物を返します。
        return best_id
    # 距離の候補がなくても短時間追跡が一意なら採用します。
    if temporal_id is not None:
        # 追跡先の人物IDを返します。
        return temporal_id
    # 新しい人物へランダムな一意IDを付けます。
    identifier = str(uuid4())
    # 表示名のない人物を顔写真だけで作ります。
    new_people.append(PersonRecord(id=identifier, name="", thumbnail_path=thumbnail_path))
    # 新しく作った人物IDを返します。
    return identifier


# 一区間の顔を旧結果へ対応させ、人物分類と保存対象を返します。
def classify_faces(samples: list["FaceSample"], video: VideoRecord, existing_faces: list[FaceRecord], people: list[PersonRecord], thumbnail_dir: Path, completed_second: float, duration: float) -> tuple[list[FaceRecord], list[PersonRecord], list[str]]:
    # サムネイルの書き込み先がなければ作ります。
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    # この動画に保存済みの顔だけを選びます。
    previous = [face for face in existing_faces if face.video_id == video.id]
    # 再解析の開始秒より少し早い実フレームを含めます。
    lower = max(0.0, video.last_analyzed_second - 0.25)
    # 最後の区間では動画末尾まで、それ以外では次区間の顔を避けて区切ります。
    upper = duration + 0.25 if completed_second >= duration else completed_second - 0.25
    # 今回処理した区間の旧顔だけを置き換えます。
    old_region = [face for face in previous if lower <= face.second < upper]
    # 旧顔のIDを保存側へ削除対象として渡します。
    removed_ids = [face.id for face in old_region]
    # 顔が見つからない区間でも旧検出を消して再解析結果へ合わせます。
    if not samples:
        # 顔のない区間から人物は新しく作りません。
        return [], [], removed_ids
    # 同じ実フレーム時刻の旧顔をまとめます。
    previous_by_frame: dict[int, list[FaceRecord]] = defaultdict(list)
    # 旧顔を六百分の一秒単位で索引化します。
    for face in old_region:
        # 同時刻の複数人も一つの一覧に保ちます。
        previous_by_frame[round(face.second * 600)].append(face)
    # 実在する人物IDだけを自動分類の候補にします。
    existing_person_ids = {person.id for person in people}
    # 各人物の多様な代表顔を保持します。
    representatives: dict[str, list[FaceRecord]] = {}
    # 置き換え対象以外の分類済み顔から比較用代表を作ります。
    for face in existing_faces:
        # 有効な人物へ分類済みで、今回置き換えない顔を選びます。
        if face.id not in removed_ids and face.person_id in existing_person_ids and (face.video_id != video.id or face.second < video.last_analyzed_second):
            # 候補となる顔を人物の代表一覧へ加えます。
            _add_representative(face, representatives)
    # 同じフレームで同一人物を二度割り当てないための索引です。
    occupied_by_frame: dict[int, set[str]] = defaultdict(set)
    # 旧顔の手動分類を含めてフレームの占有情報を作ります。
    for face in old_region:
        # 除外されず人物が決まった旧顔だけを扱います。
        if not face.excluded and face.person_id is not None:
            # 旧顔が映ったフレームの人物を記録します。
            occupied_by_frame[round(face.second * 600)].add(face.person_id)
    # 同じ動画の直近フレームで見つけた人物を保持します。
    recent: dict[str, FaceRecord] = {}
    # 追跡の最大時間差を通常・詳細モードに応じて決めます。
    maximum_gap = max(4.5, video.sample_interval * 1.5)
    # 区間開始前の顔から各人物の最後の出現を取得します。
    for face in previous:
        # 新しい区間以前で比較可能な分類済み顔だけを選びます。
        if face.id not in removed_ids and face.person_id in existing_person_ids and not face.excluded and face.bounding_box is not None and face.embedding is not None and face.second < video.last_analyzed_second and video.last_analyzed_second - face.second <= maximum_gap + 0.25:
            # 同一人物のより新しい出現だけを追跡に使います。
            if face.person_id not in recent or face.second > recent[face.person_id].second:
                # 今の顔をその人物の追跡起点にします。
                recent[face.person_id] = face
    # 同じ時刻に検出された顔をまとめます。
    samples_by_frame: dict[int, list[FaceSample]] = defaultdict(list)
    # 解析された顔を時刻の索引へ追加します。
    for sample in samples:
        # 実フレーム時刻を六百分の一秒へ丸めてまとめます。
        samples_by_frame[round(sample.second * 600)].append(sample)
    # 今回分類した顔を蓄積します。
    new_faces: list[FaceRecord] = []
    # 今回作った人物を蓄積します。
    new_people: list[PersonRecord] = []
    # 一つの旧顔を複数の新顔へ再利用しないための集合です。
    matched_old_ids: set[str] = set()
    # 入力された各顔を時刻順に慎重に分類します。
    for sample in sorted(samples, key=lambda item: item.second):
        # 実フレーム時刻を旧顔との対応付けに使います。
        frame_key = round(sample.second * 600)
        # 同時刻でまだ使われていない旧顔の候補を集めます。
        old_candidates = [face for face in previous_by_frame[frame_key] if face.id not in matched_old_ids]
        # 顔枠の重なりを主な基準に旧顔を順位付けします。
        matched = [(1.0 - _overlap(face.bounding_box, sample.bounding_box), face) for face in old_candidates if _overlap(face.bounding_box, sample.bounding_box) >= 0.3]
        # 旧形式で顔枠のない顔だけ特徴量で再利用を検討します。
        matched.extend((1.0 + _distance(face.embedding, sample.embedding), face) for face in old_candidates if face.bounding_box is None and _distance(face.embedding, sample.embedding) <= 0.25)
        # スコアが同点でもFaceRecord同士を比較しないようキーを指定します。
        reusable = min(matched, key=lambda item: item[0])[1] if matched else None
        # 旧顔を再利用する場合はIDを保持します。
        identifier = reusable.id if reusable is not None else str(uuid4())
        # 同じ旧顔が次の検出顔に選ばれないよう記録します。
        if reusable is not None:
            # 再利用済みIDの集合に入れます。
            matched_old_ids.add(reusable.id)
        # 顔の保存先をIDから決めます。
        image_path = thumbnail_dir / f"face-{identifier}.jpg"
        # 解析器が返したJPEGをローカルに保存します。
        image_path.write_bytes(sample.thumbnail_jpeg)
        # 新規顔に限り短時間追跡の候補を選びます。
        temporal_id = None if reusable is not None else _temporal_person(sample, samples_by_frame[frame_key], recent, occupied_by_frame[frame_key], maximum_gap)
        # 手動変更済みを含む旧顔の人物割当をそのまま保持します。
        person_id = reusable.person_id if reusable is not None else _cluster_person(sample, representatives, occupied_by_frame[frame_key], temporal_id, str(image_path), new_people)
        # 今回の時刻、画像、人物割当を一件の顔レコードにします。
        face = FaceRecord(id=identifier, video_id=video.id, second=sample.second, bounding_box=sample.bounding_box, person_id=person_id, thumbnail_path=str(image_path), embedding=sample.embedding, manual_assignment=reusable.manual_assignment if reusable is not None else False, excluded=reusable.excluded if reusable is not None else False)
        # 保存対象の顔一覧へ追加します。
        new_faces.append(face)
        # 割当が確かな顔だけ次の顔の候補へ加えます。
        if person_id is not None and not face.excluded:
            # 同じフレームの別顔が同じ人物を選ばないようにします。
            occupied_by_frame[frame_key].add(person_id)
            # 新しい向きの顔なら代表に加えます。
            _add_representative(face, representatives)
            # この人物の最も新しい出現として追跡に使います。
            recent[person_id] = face
    # 呼び出し側が人物を先に保存し、旧顔を削除してから新顔を保存します。
    return new_faces, new_people, removed_ids


# 自動統合の比較に使う顔特徴量と動画情報をまとめます。
@dataclass
class _Group:
    # 人物レコードの文字列IDです。
    identifier: str
    # 時間的に分散した最大三十二枚の特徴量です。
    samples: list[list[float]]
    # 最終的な統合根拠に使う全顔の特徴量です。
    all_samples: list[list[float]]
    # 全顔の特徴量の平均です。
    center: list[float]
    # 平均から最も遠い顔までの距離です。
    radius: float
    # この人物が現れた動画のIDです。
    video_ids: set[str]
    # 同時に出現した別人物を見分ける動画と十分の一秒の組です。
    frames: set[tuple[str, int]]
    # 統合時に多い側を残すための顔の件数です。
    face_count: int


# 二組を統合できるか決める根拠を保持します。
@dataclass
class _Evidence:
    # 二組間で最も近い顔の距離です。
    minimum: float
    # 一組目から二組目へ近い顔の件数です。
    first_support: int
    # 二組目から一組目へ近い顔の件数です。
    second_support: int
    # 一組目で近い顔が占める割合です。
    first_fraction: float
    # 二組目で近い顔が占める割合です。
    second_fraction: float

    # 同じ動画に映る組へは、より厳しい条件を適用します。
    def accepts(self, shared_video: bool) -> bool:
        # 同じ動画では両方向に十件、別動画では五件を要求します。
        minimum_count = 10 if shared_video else 5
        # 同じ動画では四分の一、別動画では一割の支持を要求します。
        minimum_fraction = 0.25 if shared_video else 0.10
        # 同じ動画ではより近い顔を必要とします。
        strong_distance = 0.18 if shared_video else 0.20
        # 五条件すべてを満たした場合だけ統合を認めます。
        return self.first_support >= minimum_count and self.second_support >= minimum_count and self.first_fraction >= minimum_fraction and self.second_fraction >= minimum_fraction and self.minimum <= strong_distance


# 特徴量を長さ一にそろえ、距離の比較を安定させます。
def _unit(embedding: list[float] | None, dimension: int) -> list[float] | None:
    # 特徴量の欠落や次元違いを拒否します。
    if embedding is None or len(embedding) != dimension:
        # 比較できない特徴量は候補から外します。
        return None
    # 全成分の二乗和から長さを求めます。
    length = math.sqrt(math.fsum(value * value for value in embedding))
    # ゼロや不正値なら特徴量を利用しません。
    if not math.isfinite(length) or length == 0:
        # 不正な特徴量は人物の証拠にしません。
        return None
    # 全成分を長さで割った単位ベクトルを返します。
    return [value / length for value in embedding]


# 時間順の顔から全体に広がる件数を選びます。
def _evenly_spaced(records: list[FaceRecord], count: int) -> list[FaceRecord]:
    # 空の入力や零件指定では顔を返しません。
    if not records or count <= 0:
        # 選択結果は空です。
        return []
    # 指定数が顔の件数以上ならすべて選びます。
    if count >= len(records):
        # 入力された顔をそのまま返します。
        return records[:]
    # 一件だけなら時系列の中央を代表にします。
    if count == 1:
        # 中央付近の顔を一件返します。
        return [records[len(records) // 2]]
    # 最初から最後まで均等に分布する添字を使います。
    return [records[round(index * (len(records) - 1) / (count - 1))] for index in range(count)]


# 大きな動画に偏らないよう各動画の複数時刻から顔を選びます。
def _balanced_faces(records: list[FaceRecord], limit: int) -> list[FaceRecord]:
    # 動画IDごとに顔を集めます。
    by_video: dict[str, list[FaceRecord]] = defaultdict(list)
    # 各顔を所属動画の一覧へ追加します。
    for face in records:
        # 動画IDをキーにして一件加えます。
        by_video[face.video_id].append(face)
    # 各動画へできるだけ同じ件数の枠を与えます。
    quota = max(1, limit // max(1, len(by_video)))
    # 選択された顔を蓄積します。
    selected: list[FaceRecord] = []
    # 動画IDを並べて処理順を安定させます。
    for video_id in sorted(by_video):
        # この動画の顔を時刻順にします。
        ordered = sorted(by_video[video_id], key=lambda face: face.second)
        # 動画ごとの枠を時刻全体へ分散させます。
        selected.extend(_evenly_spaced(ordered, min(quota, len(ordered))))
    # 同じ顔を空き枠で二度選ばないようにします。
    selected_ids = {face.id for face in selected}
    # まだ選ばれていない顔を時刻順に集めます。
    remainder = sorted((face for face in records if face.id not in selected_ids), key=lambda face: face.second)
    # 空き枠へ残りの時刻の顔を分散して入れます。
    selected.extend(_evenly_spaced(remainder, max(0, limit - len(selected))))
    # 最大件数以内の顔を返します。
    return selected[:limit]


# 二組の全顔を比べて、互いに近い顔の件数を数えます。
def _compare_groups(first: _Group, second: _Group) -> _Evidence | None:
    # 一組目の各顔から二組目への最短距離を無限大で初期化します。
    nearest_first = [NO_MATCH] * len(first.all_samples)
    # 二組目の各顔から一組目への最短距離を無限大で初期化します。
    nearest_second = [NO_MATCH] * len(second.all_samples)
    # 二組全体で最も近い距離を記録します。
    minimum = NO_MATCH
    # 一組目の各顔を順に見ます。
    for first_index, left in enumerate(first.all_samples):
        # 二組目の各顔と比較します。
        for second_index, right in enumerate(second.all_samples):
            # 単位ベクトル同士の距離を計算します。
            distance = max(0.0, 1.0 - math.fsum(a * b for a, b in zip(left, right)))
            # 一組目から見た最短距離を更新します。
            nearest_first[first_index] = min(nearest_first[first_index], distance)
            # 二組目から見た最短距離を更新します。
            nearest_second[second_index] = min(nearest_second[second_index], distance)
            # 二組全体の最短距離を更新します。
            minimum = min(minimum, distance)
    # 最も近い一組も0.20以内にない場合は統合を検討しません。
    if minimum > 0.20:
        # 根拠のない組として返します。
        return None
    # 一組目で相手に近い顔を数えます。
    first_support = sum(distance <= 0.25 for distance in nearest_first)
    # 二組目で相手に近い顔を数えます。
    second_support = sum(distance <= 0.25 for distance in nearest_second)
    # 双方向で少なくとも五件の根拠を必要とします。
    if first_support < 5 or second_support < 5:
        # 単発の似た顔では自動統合しません。
        return None
    # 距離、件数、割合を一つの根拠にまとめます。
    return _Evidence(minimum, first_support, second_support, first_support / len(nearest_first), second_support / len(nearest_second))


# 自動統合先を求め、元人物IDから統合先IDへの対応を返します。
def reconcile_groups(faces: list[FaceRecord], videos: list[VideoRecord], people: list[PersonRecord]) -> dict[str, str]:
    # 解析完了した動画だけを根拠にします。
    ready_video_ids = {video.id for video in videos if video.state == "ready"}
    # 一件でも手動修正された人物グループ全体を保護します。
    protected_ids = {face.person_id for face in faces if face.manual_assignment and face.person_id is not None}
    # 実在する人物だけを候補にします。
    existing_ids = {person.id for person in people}
    # 候補となる顔を人物IDごとに集めます。
    by_person: dict[str, list[FaceRecord]] = defaultdict(list)
    # 顔の状態と人物IDを一件ずつ調べます。
    for face in faces:
        # 除外顔、未解析動画、手動群、特徴量なし、孤立した人物IDを除きます。
        if not face.excluded and face.video_id in ready_video_ids and face.person_id in existing_ids and face.person_id not in protected_ids and face.embedding is not None:
            # この顔を人物の比較情報へ追加します。
            by_person[face.person_id].append(face)
    # 条件を満たした人物の比較用情報を保持します。
    groups: list[_Group] = []
    # 人物IDの順序を固定して結果を安定させます。
    for identifier in sorted(by_person):
        # この人物の全顔を取り出します。
        records = by_person[identifier]
        # 動画と時刻に分散させた顔を最大三十二件選びます。
        selected = _balanced_faces(records, 32)
        # 五件未満の人物は双方向の根拠を作れません。
        if len(selected) < 5:
            # この人物を自動統合の候補から外します。
            continue
        # 最初の顔の特徴量次元を基準にします。
        dimension = len(selected[0].embedding or [])
        # 欠損または異なる次元が混じるグループを除外します。
        if dimension == 0 or any(face.embedding is None or len(face.embedding) != dimension for face in selected):
            # 異なるモデルの値を同じ人物の証拠にしません。
            continue
        # 全顔を正規化し、無効な顔だけ取り除きます。
        all_samples = [unit for face in records if (unit := _unit(face.embedding, dimension)) is not None]
        # 代表顔も同じ規則で正規化します。
        sample_units = [unit for face in selected if (unit := _unit(face.embedding, dimension)) is not None]
        # 比較できる代表が五件未満なら候補から外します。
        if len(sample_units) < 5 or not all_samples:
            # 不正な数値を人物の証拠にしません。
            continue
        # 各次元の全顔平均を作ります。
        center = [math.fsum(sample[index] for sample in all_samples) / len(all_samples) for index in range(dimension)]
        # 平均から最も遠い顔までのユークリッド距離を求めます。
        radius = max(math.dist(sample, center) for sample in all_samples)
        # 同じ動画に映った候補へ厳しい条件を適用するため記録します。
        video_ids = {face.video_id for face in records}
        # 同時出現を判定する時刻を十分の一秒単位で記録します。
        frames = {(face.video_id, round(face.second * 10)) for face in records}
        # 一人分の比較情報を一覧へ追加します。
        groups.append(_Group(identifier, sample_units, all_samples, center, radius, video_ids, frames, len(records)))
    # 二人未満なら統合先はありません。
    if len(groups) < 2:
        # 変更なしの対応表を返します。
        return {}
    # 自動統合が認められた人物ペアと根拠を保持します。
    edges: list[tuple[str, str, _Evidence]] = []
    # 連鎖統合の際に元グループ間の全組を確認する辞書です。
    evidence_by_pair: dict[frozenset[str], _Evidence] = {}
    # 各人物ペアを一度ずつ比較します。
    for first_index, first in enumerate(groups):
        # 二人目は一人目より後の人物から選びます。
        for second in groups[first_index + 1:]:
            # 同じフレームに二人が映る場合は別人として保護します。
            if first.frames & second.frames:
                # 自動統合の候補から外します。
                continue
            # 次元違いの特徴量は比較しません。
            if len(first.center) != len(second.center):
                # モデルの違う人物同士を比べません。
                continue
            # 平均と半径から近い顔が存在し得ない組を省きます。
            if math.dist(first.center, second.center) - first.radius - second.radius > math.sqrt(0.5):
                # 距離の下界だけで不一致と分かる組を飛ばします。
                continue
            # 少数の代表顔に0.35以内の手掛かりがあるか調べます。
            coarse = any(_distance(left, right) <= 0.35 for left in first.samples for right in second.samples)
            # 粗い手掛かりもない組を全顔比較しません。
            if not coarse:
                # この組を候補から外します。
                continue
            # 双方向の全顔比較から統合の根拠を得ます。
            evidence = _compare_groups(first, second)
            # 双方向の支持数を満たさない組は除外します。
            if evidence is None:
                # 弱い根拠では人物を混ぜません。
                continue
            # 同じ動画にも現れるか判定します。
            shared_video = bool(first.video_ids & second.video_ids)
            # 同じ動画に映る場合の追加条件も含めて確認します。
            if evidence.accepts(shared_video):
                # 統合候補を優先順位付け用の一覧へ加えます。
                edges.append((first.identifier, second.identifier, evidence))
                # 元グループの組の根拠を保存します。
                evidence_by_pair[frozenset((first.identifier, second.identifier))] = evidence
    # 元の人物ごとの動画集合を索引化します。
    original_videos = {group.identifier: group.video_ids for group in groups}
    # 跨動画の一致を先にし、その後に支持率が強い一致を扱います。
    edges.sort(key=lambda edge: (bool(original_videos[edge[0]] & original_videos[edge[1]]), -(min(edge[2].first_fraction, edge[2].second_fraction) + 0.1 * (1 - edge[2].minimum))))
    # 各人物を最初は独立した統合集合として扱います。
    parent = {group.identifier: group.identifier for group in groups}
    # 統合集合の動画を保持します。
    component_videos = {group.identifier: set(group.video_ids) for group in groups}
    # 統合集合の出現フレームを保持します。
    component_frames = {group.identifier: set(group.frames) for group in groups}
    # 統合先は顔件数の多い人物にするため件数を保持します。
    component_sizes = {group.identifier: group.face_count for group in groups}
    # 連鎖統合前の元グループを各集合に保持します。
    component_members = {group.identifier: {group.identifier} for group in groups}
    # 強い根拠がある各組を順番に検討します。
    for first_id, second_id, evidence in edges:
        # 一人目の現在の統合先を親の連鎖から探します。
        first_root = _root(first_id, parent)
        # 二人目の現在の統合先を親の連鎖から探します。
        second_root = _root(second_id, parent)
        # すでに同じ人物になっていれば何もしません。
        if first_root == second_root:
            # 次の候補へ進みます。
            continue
        # 統合集合内に同時出現する顔があれば別人として守ります。
        if component_frames[first_root] & component_frames[second_root]:
            # 同時出現した二組を結びません。
            continue
        # 統合集合で同じ動画を共有しているか確認します。
        shared_video = bool(component_videos[first_root] & component_videos[second_root])
        # 現在の集合関係で厳しい条件に届かない場合は飛ばします。
        if not evidence.accepts(shared_video):
            # 集合を統合せず次の組へ進みます。
            continue
        # 二集合に含まれる全ての元グループ間で独立した根拠を要求します。
        all_pairs_supported = all((pair_evidence := evidence_by_pair.get(frozenset((left, right)))) is not None and pair_evidence.accepts(shared_video) for left in component_members[first_root] for right in component_members[second_root])
        # 一組だけの橋渡しでは別人を連鎖統合しません。
        if not all_pairs_supported:
            # 二集合を分けたまま次へ進みます。
            continue
        # 顔件数が多い側の人物IDを残します。
        keep = first_root if component_sizes[first_root] >= component_sizes[second_root] else second_root
        # もう一方の人物IDを統合元とします。
        remove = second_root if keep == first_root else first_root
        # 統合元の最終的な親を記録します。
        parent[remove] = keep
        # 両集合が映る動画を統合先へ加えます。
        component_videos[keep].update(component_videos[remove])
        # 両集合の出現フレームを統合先へ加えます。
        component_frames[keep].update(component_frames[remove])
        # 統合先の顔件数を増やします。
        component_sizes[keep] += component_sizes[remove]
        # 元グループの全IDを統合先へ加えます。
        component_members[keep].update(component_members[remove])
    # 最終的な統合元と統合先の対応を作ります。
    mapping: dict[str, str] = {}
    # 元人物ごとに親の連鎖をたどります。
    for group in groups:
        # 統合後の人物IDを取得します。
        destination = _root(group.identifier, parent)
        # 自分自身ではない先に移った人物だけを保存します。
        if destination != group.identifier:
            # 元人物IDから統合先IDへの対応を記録します。
            mapping[group.identifier] = destination
    # SQLiteは変更せず、呼び出し側がトランザクションで適用する対応を返します。
    return mapping


# 統合された人物の親をたどって最終的な人物IDを返します。
def _root(identifier: str, parent: dict[str, str]) -> str:
    # 指定された元人物からたどり始めます。
    current = identifier
    # 親が自分自身になるまで統合先をたどります。
    while parent[current] != current:
        # 一段上の統合先へ進みます。
        current = parent[current]
    # 最後に残る人物IDを返します。
    return current
