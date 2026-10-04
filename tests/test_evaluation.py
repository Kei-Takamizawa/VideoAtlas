import unittest
from videoatlas.evaluation import evaluate_pairs, optimize_thresholds


class EvaluationTests(unittest.TestCase):
    def test_false_positive_denominators_and_auc(self):
        examples = [{"same": True, "score": .9}, {"same": True, "score": .6}, {"same": False, "score": .8}, {"same": False, "score": .2}, {"same": False, "score": .1}]
        report = evaluate_pairs(examples, .7)
        self.assertEqual((report["tp"], report["fp"], report["tn"], report["fn"]), (1, 1, 2, 1))
        self.assertEqual(report["false_positive_rate"], 1 / 3)
        self.assertEqual(report["false_discovery_fraction"], .5)
        self.assertAlmostEqual(report["auc"], 5 / 6)

    def test_undefined_denominators_and_ties(self):
        report = evaluate_pairs([{"same": True, "score": .5}, {"same": False, "score": .5}], .9)
        self.assertIsNone(report["precision"])
        self.assertEqual(report["auc"], .5)
        self.assertIsNone(evaluate_pairs([{"same": True, "score": 1}], .5)["false_positive_rate"])

    def test_missing_ensemble_evidence_is_counted(self):
        examples = [{"same": True, "scores": {"face01:v1": .9}}, {"same": False, "scores": {"face01:v1": .2, "adaface:v1": .3}}]
        report = evaluate_pairs(examples, .7, {"face01:v1": .5, "adaface:v1": .5})
        self.assertEqual(report["excluded_missing_evidence"], 1)
        self.assertEqual(report["sample_count"], 1)

    def test_optimizer_only_reports_candidates(self):
        examples = [{"same": True, "scores": {"face01:v1": .9, "adaface:v1": .8}}, {"same": False, "scores": {"face01:v1": .8, "adaface:v1": .1}}]
        report = optimize_thresholds(examples, [.7, .85])
        self.assertTrue(report["candidate_only"])
        self.assertTrue(report["requires_held_out_validation"])
        self.assertEqual(report["best_candidate"]["fp"], 0)
        self.assertEqual(report["best_candidate"]["fn"], 0)
        self.assertGreater(len(report["candidates"]), 2)

    def test_versions_are_not_ensemble_candidates(self):
        examples = [{"same": True, "scores": {"face01:v1": .9, "face01:v2": .8}}, {"same": False, "scores": {"face01:v1": .2, "face01:v2": .1}}]
        report = optimize_thresholds(examples, [.7])
        self.assertTrue(all(len(candidate["weights"]) == 1 for candidate in report["candidates"]))
        with self.assertRaises(ValueError):
            optimize_thresholds(examples, [.7], [{"face01:v1": .5, "face01:v2": .5}])


if __name__ == "__main__":
    unittest.main()
