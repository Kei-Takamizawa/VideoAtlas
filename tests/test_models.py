from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading
from collections import OrderedDict

import cv2
import numpy as np
import pytest

from videoatlas.analyzer import FaceSample, VideoAnalyzer, _SessionState
from videoatlas.model_adapters import (ADAFACE_TEMPLATE, SCRFDDetector, OnnxRunner,
                                      adaface_tensor, face01_tensor, align_crop, similarity_transform)
from videoatlas.recognition import RecognitionOptions
from videoatlas import model_adapters


def test_independent_channel_orders_and_normalization():
    primary = np.full((224,224,3),[0,128,255],np.uint8)
    auxiliary = np.full((112,112,3),[0,128,255],np.uint8)
    np.testing.assert_allclose(face01_tensor(primary)[:,0,0],[(1-.485)/.229,(128/255-.456)/.224,(0-.406)/.225],rtol=1e-6)
    np.testing.assert_allclose(adaface_tensor(auxiliary)[:,0,0],[-1,128/127.5-1,1],atol=1e-7)


def test_transform_preserves_all_landmarks_without_reflection():
    source = ADAFACE_TEMPLATE*2+25
    matrix = similarity_transform(source,ADAFACE_TEMPLATE)
    np.testing.assert_allclose(cv2.transform(source[None],matrix)[0],ADAFACE_TEMPLATE,atol=1e-5)
    assert np.linalg.det(matrix[:,:2]) > 0


def test_degenerate_landmarks_fail():
    with pytest.raises(ValueError):
        similarity_transform(np.ones((5,2)),ADAFACE_TEMPLATE)
    with pytest.raises(ValueError):
        align_crop(np.zeros((200,200,3),np.uint8),np.full((5,2),np.nan),'face01')


def detector_outputs():
    detector = SCRFDDetector.__new__(SCRFDDetector)
    detector.options = RecognitionOptions()
    detector.size,detector.strides,detector.anchors = (128,128),(8,16,32),2
    counts = [(128//stride)**2*2 for stride in detector.strides]
    scores = [np.zeros((count,1),np.float32) for count in counts]
    boxes = [np.zeros((count,4),np.float32) for count in counts]
    points = [np.zeros((count,10),np.float32) for count in counts]
    # grid point (5,5) at stride 8 -> anchor (40,40); two duplicate boxes.
    index = (5*16+5)*2
    scores[0][index:index+2] = [[.9],[.8]]
    boxes[0][index:index+2] = [2,2,2,2]
    points[0][index:index+2] = np.tile([-1,-1,1,-1,0,0,-1,1,1,1],(2,1))
    return detector,scores+boxes+points


def test_scrfd_decoding_scaling_and_nms():
    detector,outputs = detector_outputs()
    detections = detector.decode(outputs,.5,256,256)
    assert len(detections) == 1
    assert detections[0].box == (48,48,112,112)
    np.testing.assert_allclose(detections[0].landmarks,[[64,64],[96,64],[80,80],[64,96],[96,96]])


def test_scrfd_rejects_nonfinite_output():
    detector,outputs = detector_outputs()
    outputs[3][0,0] = np.nan
    with pytest.raises(ValueError,match='geometry'):
        detector.decode(outputs,1,128,128)


def test_gpu_execution_retry_is_cpu_once():
    runner = OnnxRunner.__new__(OnnxRunner)
    class Failed:
        def run(self,*args):
            raise RuntimeError('GPU allocation failed')
        def get_providers(self):
            return ['CUDAExecutionProvider','CPUExecutionProvider']
    class CPU:
        def run(self,*args):
            return [np.asarray([[1.]])]
        def get_providers(self):
            return ['CPUExecutionProvider']
    runner.session,runner.input,runner.outputs = Failed(),SimpleNamespace(name='image'),[SimpleNamespace(name='result')]
    runner.lock = threading.RLock()
    runner.cpu_session = lambda: CPU()
    assert runner.run(np.zeros(1))[0][0,0] == 1
    assert runner.session.get_providers() == ['CPUExecutionProvider']


def test_cpu_execution_errors_propagate():
    runner = OnnxRunner.__new__(OnnxRunner)
    runner.session = SimpleNamespace(run=lambda *args: (_ for _ in ()).throw(RuntimeError('bad model')),get_providers=lambda:['CPUExecutionProvider'])
    runner.input,runner.outputs = SimpleNamespace(name='image'),[SimpleNamespace(name='result')]
    runner.lock = threading.RLock()
    with pytest.raises(RuntimeError,match='bad model'):
        runner.run(np.zeros(1))


def test_config_bounds_and_fingerprint(tmp_path):
    options = RecognitionOptions()
    options.validate()
    assert not options.auto_merge_enabled
    assert replace(options,cache_dir=tmp_path,database_dir=tmp_path,log_level='DEBUG').analysis_fingerprint() == options.analysis_fingerprint()
    assert replace(options,detection_threshold=.7).analysis_fingerprint() != options.analysis_fingerprint()
    for changes in ({'min_quality':float('nan')},{'min_samples':1},{'adaface_enabled':True},{'medium_threshold':.9}):
        with pytest.raises(ValueError):
            replace(options,**changes).validate()


def test_json_relative_paths_and_extensions(tmp_path):
    config = tmp_path/'recognition.json'
    config.write_text('{"detection_model_path":"models/scrfd.onnx","extensions":[".mp4"],"recursive":false}')
    options = RecognitionOptions.load(config)
    assert options.detection_model_path == tmp_path/'models/scrfd.onnx'
    assert options.extensions == ('.mp4',)
    assert not options.recursive


def test_small_and_blurry_faces_never_receive_embeddings():
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.options = RecognitionOptions()
    points = ADAFACE_TEMPLATE*2+100
    frame = np.full((400,400,3),128,np.uint8)
    assert analyzer._prepare_face(frame,points,(0,0,30,30),.95)[3] == 'face_too_small'
    result = analyzer._prepare_face(frame,points,(100,100,300,330),.95)
    assert result[0] is None and result[3] == 'contrast_too_low'


def test_cached_recalculation_uses_new_model_only(tmp_path):
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.options,analyzer._closed,analyzer._adaface = RecognitionOptions(cache_dir=tmp_path),False,None
    analyzer._lifecycle_lock,analyzer._active = threading.RLock(),False
    analyzer.embedding_model = 'face01:new-version'
    analyzer._infer_batch = lambda tensors,stop: [[1.,0.]]*len(tensors)
    path = analyzer._cache_crops(np.full((224,224,3),128,np.uint8),np.full((112,112,3),128,np.uint8))
    sample = FaceSample(1,(0,0,.5,.5),b'face',[0.,1.],'face01:old',aligned_path=path,embeddings={'face01:old':[0.,1.]})
    result = analyzer.recalculate_samples([sample])[0]
    assert result.embeddings == {'face01:new-version':[1.,0.]}
    assert sample.embedding_model == 'face01:old'
    assert Path(path).with_suffix('.adaface.png').is_file()


def test_missing_cache_is_explicit_failure(tmp_path):
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer._closed,analyzer._adaface = False,None
    analyzer._lifecycle_lock,analyzer._active = threading.RLock(),False
    sample = FaceSample(1,(0,0,.5,.5),b'face',None,aligned_path=str(tmp_path/'missing.png'))
    with pytest.raises(FileNotFoundError):
        analyzer.recalculate_samples([sample])


def test_corrupt_cached_crop_cannot_be_reused(tmp_path):
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.options,analyzer._closed,analyzer._adaface = RecognitionOptions(cache_dir=tmp_path),False,None
    analyzer._lifecycle_lock,analyzer._active = threading.RLock(),False
    path = analyzer._cache_crops(np.full((224,224,3),128,np.uint8),np.full((112,112,3),128,np.uint8))
    Path(path).write_bytes(b'corrupted')
    sample = FaceSample(1,(0,0,.5,.5),b'face',None,aligned_path=path)
    with pytest.raises(ValueError,match='checksum'):
        analyzer.recalculate_samples([sample])
    assert not analyzer._active


def test_face01_gpu_failure_restarts_with_distinct_cpu_model_key():
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.options,analyzer._closed = RecognitionOptions(),False
    model_input = SimpleNamespace(name='images',shape=[1,3,224,224],type='tensor(float)')
    model_output = SimpleNamespace(name='embedding',shape=[1,512],type='tensor(float)')
    class CPU:
        def get_inputs(self):
            return [model_input]
        def get_outputs(self):
            return [model_output]
        def disable_fallback(self):
            pass
        def run(self,*args):
            return [np.ones((1,512),np.float32)]
    analyzer._ort = SimpleNamespace(SessionOptions=lambda:SimpleNamespace(),InferenceSession=lambda *args,**kwargs:CPU(),__version__='test')
    analyzer._state = _SessionState(None,model_input,model_output,False,('CUDAExecutionProvider','CPUExecutionProvider'),'gpu-profile',threading.Lock())
    analyzer._input,analyzer._output,analyzer._dynamic_batch = model_input,model_output,False
    analyzer._gpu_buffers = OrderedDict()
    analyzer.model_path,analyzer.model_sha256 = Path('fake.onnx'),'artifact'
    analyzer.embedding_model,analyzer.runtime_description = 'face01:gpu','test'
    result = analyzer._infer_batch([np.zeros((3,224,224),np.float32)])
    assert result is not None and len(result[0]) == 512
    assert abs(np.linalg.norm(result[0])-1) < 1e-6
    assert 'cpu-fp32' in analyzer.embedding_model
    assert analyzer._state.providers == ('CPUExecutionProvider',)


@pytest.fixture
def sampling_fixture(tmp_path):
    """Real tiny MJPEG stream; replace only face inference to inspect scheduling."""
    path = tmp_path/'sampling.avi'
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20,(80,80))
    assert writer.isOpened()
    for index in range(40):
        writer.write(np.full((80,80,3),index*5,np.uint8))
    writer.release()
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer._closed,analyzer._active = False,False
    analyzer._lifecycle_lock = threading.RLock()
    analyzer.options = RecognitionOptions(face_sample_interval=.5)
    analyzer._CHUNK_SIZE = 2
    analyzer._process_frame = lambda frame,second,stop,poster: ([FaceSample(second,(0,0,1,1),b'face',None)],b'poster' if poster else None)
    return analyzer,path


def test_forced_timestamps_merge_with_regular_sampling_without_timestamp_duplicates(sampling_fixture):
    analyzer,path = sampling_fixture
    chunks = list(analyzer.iter_chunks(path,.5,end=1,forced_times=[.75,.25,.35,.35,.5,1,2,3]))
    faces = [face for chunk in chunks for face in chunk.faces]
    assert [face.second for face in faces] == [0,.25,.35,.5,.75]
    assert [face.frame_number for face in faces] == [0,5,7,10,15]
    assert [chunk.completed_second for chunk in chunks] == [.35,.75,1]
    assert not analyzer._active


def test_resume_preserves_pending_anchor_and_zero_based_regular_grid(sampling_fixture):
    analyzer,path = sampling_fixture
    stream = analyzer.iter_chunks(path,.5,end=1,forced_times=[.25,.35,.75])
    first = next(stream)
    assert first.completed_second == .35
    first_times = [sample.second for sample in first.faces]
    stream.close()
    resumed = list(analyzer.iter_chunks(path,.5,start=first.completed_second,end=1,forced_times=[.25,.35,.75]))
    assert first_times+[face.second for chunk in resumed for face in chunk.faces] == [0,.25,.35,.5,.75]
    assert [face.frame_number for chunk in resumed for face in chunk.faces] == [7,10,15]
    assert not analyzer._active


def test_decimal_grid_collision_keeps_exact_review_anchor_once(sampling_fixture):
    analyzer,path = sampling_fixture
    chunks = list(analyzer.iter_chunks(path,.2,end=1,forced_times=[.6]))
    faces = [face for chunk in chunks for face in chunk.faces]
    assert [face.second for face in faces] == [0,.2,.4,.6,.8]
    assert [face.frame_number for face in faces] == [0,4,8,12,16]


def test_cancel_after_chunk_does_not_consume_pending_anchor(sampling_fixture):
    analyzer,path = sampling_fixture
    stopped = False
    stream = analyzer.iter_chunks(path,.5,end=1,forced_times=[.25,.35],stop=lambda:stopped)
    first = next(stream)
    stopped = True
    cancelled = next(stream)
    assert first.completed_second == cancelled.completed_second == .35
    assert cancelled.cancelled and cancelled.faces == []
    with pytest.raises(StopIteration):
        next(stream)
    stopped = False
    resumed = list(analyzer.iter_chunks(path,.5,start=cancelled.completed_second,end=1,forced_times=[.25,.35]))
    assert [face.second for chunk in resumed for face in chunk.faces] == [.35,.5]


@pytest.mark.parametrize('anchors',[[float('nan')],[float('inf')],[-.01],[True],['.5'],(.5,)])
def test_invalid_forced_times_fail_before_decoder_access(anchors,sampling_fixture):
    analyzer,path = sampling_fixture
    with pytest.raises(ValueError,match='forced_times'):
        list(analyzer.iter_chunks(path,.5,forced_times=anchors))
    assert not analyzer._active


def test_windows_nvidia_bin_discovery_is_process_local_and_idempotent(tmp_path,monkeypatch):
    directory = tmp_path/'nvidia'/'cudnn'/'bin'
    directory.mkdir(parents=True)
    registered = []
    def distribution(name):
        if name != 'nvidia-cudnn-cu12':
            raise model_adapters.importlib.metadata.PackageNotFoundError(name)
        return SimpleNamespace(locate_file=lambda relative:tmp_path/relative)
    monkeypatch.setattr(model_adapters.importlib.metadata,'distribution',distribution)
    monkeypatch.setattr(model_adapters.sys,'platform','win32')
    monkeypatch.setattr(model_adapters.os,'add_dll_directory',lambda path:registered.append(path) or object())
    monkeypatch.setattr(model_adapters,'_NVIDIA_DLL_HANDLES',{})
    monkeypatch.setenv('PATH','original-system-path')
    model_adapters.configure_windows_nvidia_libraries()
    model_adapters.configure_windows_nvidia_libraries()
    assert registered == [str(directory.resolve())]
    assert model_adapters.os.environ['PATH'].split(model_adapters.os.pathsep) == [str(directory.resolve()),'original-system-path']


def collect_frame_times(analyzer,path,**kwargs):
    sampled = []
    infer = analyzer._process_frame
    def inspect(frame,second,stop,poster):
        sampled.append(second)
        return infer(frame,second,stop,poster)
    analyzer._process_frame = inspect
    chunks = list(analyzer.iter_chunks(path,.5,**kwargs))
    return sampled,chunks


def test_no_face_frames_keep_regular_rate_and_anchors_do_not_activate_dense_sampling(sampling_fixture):
    analyzer,path = sampling_fixture
    analyzer.options = replace(analyzer.options,face_sample_interval=.2)
    analyzer._process_frame = lambda *args:([],None)
    sampled,chunks = collect_frame_times(analyzer,path,forced_times=[.25,.35])
    assert sampled == [0,.25,.35,.5,1,1.5]
    assert chunks[-1].completed_second == 2


def test_face_detection_activates_bounded_dense_window_without_losing_regular_events(sampling_fixture):
    analyzer,path = sampling_fixture
    analyzer.options = replace(analyzer.options,face_sample_interval=.2)
    analyzer._process_frame = lambda frame,second,stop,poster:([FaceSample(second,(0,0,1,1),b'face',None)] if second == 0 else [],None)
    sampled,chunks = collect_frame_times(analyzer,path)
    np.testing.assert_allclose(sampled,[0,.2,.4,.5,.7,.9,1,1.2,1.4,1.5],atol=1e-12)
    assert {0,.5,1,1.5}.issubset(sampled)
    assert max(np.diff(sampled)) <= .2+1e-9
    assert chunks[0].completed_second == pytest.approx(.4)
    assert chunks[-1].completed_second == 2


def test_recent_positive_frames_extend_activity_window(sampling_fixture):
    analyzer,path = sampling_fixture
    analyzer.options = replace(analyzer.options,face_sample_interval=.2)
    sampled,_ = collect_frame_times(analyzer,path)
    assert sampled[-1] == pytest.approx(1.9)
    assert max(np.diff(sampled)) <= .2+1e-9
    assert {0,.5,1,1.5}.issubset(sampled)


def test_dense_collision_preserves_exact_review_anchor_timestamp(sampling_fixture):
    analyzer,path = sampling_fixture
    analyzer.options = replace(analyzer.options,face_sample_interval=.2)
    anchor = .4000000001
    sampled,chunks = collect_frame_times(analyzer,path,end=1,forced_times=[anchor])
    assert sampled[:4] == [0,.2,anchor,.5]
    assert sum(abs(second-.4)<1e-9 for second in sampled) == 1
    assert next(face.frame_number for chunk in chunks for face in chunk.faces if face.second == anchor) == 8


def test_adaptive_cancel_resume_starts_at_pending_cursor_and_conservatively_resamples(sampling_fixture):
    analyzer,path = sampling_fixture
    analyzer.options = replace(analyzer.options,face_sample_interval=.2)
    stopped = False
    stream = analyzer.iter_chunks(path,.5,forced_times=[.35],stop=lambda:stopped)
    first = next(stream)
    assert [face.second for face in first.faces] == [0,.2]
    assert first.completed_second == .35
    stopped = True
    cancelled = next(stream)
    assert cancelled.cancelled and cancelled.completed_second == .35
    stream.close()
    # A fresh worker has no previous activity state. Even if the resume frame
    # has no face, it covers a bounded 1.5-second dense window conservatively.
    analyzer._process_frame = lambda *args:([],None)
    sampled,resumed = collect_frame_times(analyzer,path,start=cancelled.completed_second,forced_times=[.35])
    assert sampled[:3] == pytest.approx([.35,.5,.7])
    assert max(np.diff([second for second in sampled if second <= 1.85])) <= .2+1e-9
    assert resumed[-1].completed_second == 2


def test_adaptive_setting_never_slows_faster_regular_grid(sampling_fixture):
    analyzer,path = sampling_fixture
    analyzer.options = replace(analyzer.options,face_sample_interval=.3)
    sampled = []
    infer = analyzer._process_frame
    analyzer._process_frame = lambda frame,second,stop,poster:(sampled.append(second) or infer(frame,second,stop,poster))
    list(analyzer.iter_chunks(path,.1,end=.5))
    np.testing.assert_allclose(sampled,[0,.1,.2,.3,.4],atol=1e-12)
