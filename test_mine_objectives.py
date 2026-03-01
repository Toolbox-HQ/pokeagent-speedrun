"""
Integration test for mine_objectives and its helpers in models/train/train_agent.py.
Uses real data from .cache/pokeagent/online/.../agent_data.

Run from the repo root:
    python test_mine_objectives.py
"""
import json
import os
import sys
import unittest
from pathlib import Path

AGENT_DATA_DIR = ".cache/pokeagent/online/87a656a8-e16f-45c7-ae5b-d5cf7d182b8f/agent_data"
DINOV2_DIR = ".cache/pokeagent/dinov2"

JSON_FILES = sorted(
    os.path.join(AGENT_DATA_DIR, f)
    for f in os.listdir(AGENT_DATA_DIR)
    if f.endswith(".json")
)
# Keep a single-file alias for tests that don't need the full set
JSON_FILE = JSON_FILES[0]

from models.train.train_agent import (
    AgentObjectiveManager,
    ObjectivesLookup,
    filter_clusters,
    get_objective_dataset_json,
    load_objective_dataset,
    mine_objectives,
)


class TestObjectiveManager(unittest.TestCase):
    """Integration tests for AgentObjectiveManager using the real HDBSCAN clusterer."""

    @classmethod
    def setUpClass(cls):
        # Reuse the clusterer and embeddings already computed by TestMineObjectivesStats.
        cls.clusterer = TestMineObjectivesStats.clusterer
        cls.valid_ids = TestMineObjectivesStats.valid_cluster_ids
        all_embeds = TestMineObjectivesStats.per_frame_embed
        filtered_labels = TestMineObjectivesStats.filtered_labels
        # Grab up to 20 frames that are confirmed to belong to valid clusters.
        matched_indices = [
            i for i, l in enumerate(filtered_labels)
            if l in cls.valid_ids
        ][:20]
        cls.known_matched_embeds = list(all_embeds[matched_indices])

    def setUp(self):
        self.manager = AgentObjectiveManager(self.clusterer, self.valid_ids)

    # --- __init__ ---

    def test_init_achieved_objectives_empty(self):
        self.assertEqual(self.manager.achieved_objectives, [])

    def test_init_all_matched_objectives_false(self):
        for v in self.manager.matched_objectives.values():
            self.assertFalse(v)

    def test_init_matched_objectives_keys_equal_valid_ids(self):
        self.assertEqual(set(self.manager.matched_objectives.keys()), self.valid_ids)

    # --- mine_and_add_objectives ---

    def test_mine_achieves_objectives_for_known_matched_frames(self):
        self.manager.mine_and_add_objectives(self.known_matched_embeds)
        self.assertGreater(len(self.manager.achieved_objectives), 0)

    def test_mine_marks_matched_objectives_true(self):
        self.manager.mine_and_add_objectives(self.known_matched_embeds)
        matched_count = sum(1 for v in self.manager.matched_objectives.values() if v)
        self.assertGreater(matched_count, 0)

    def test_mine_each_cluster_appears_at_most_once_in_achieved(self):
        self.manager.mine_and_add_objectives(self.known_matched_embeds)
        # achieved count == number of True entries in matched_objectives
        matched_count = sum(1 for v in self.manager.matched_objectives.values() if v)
        self.assertEqual(len(self.manager.achieved_objectives), matched_count)

    def test_mine_no_new_objectives_on_repeated_call(self):
        self.manager.mine_and_add_objectives(self.known_matched_embeds)
        count_after_first = len(self.manager.achieved_objectives)
        self.manager.mine_and_add_objectives(self.known_matched_embeds)
        self.assertEqual(len(self.manager.achieved_objectives), count_after_first)

    def test_achieved_objectives_are_numpy_arrays(self):
        import numpy as np
        self.manager.mine_and_add_objectives(self.known_matched_embeds)
        for obj in self.manager.achieved_objectives:
            self.assertIsInstance(obj, np.ndarray)

    def test_achieved_objective_embed_matches_input(self):
        import numpy as np
        # Mine one frame at a time; verify each stored embed is one of the inputs.
        input_set = [e.tobytes() for e in self.known_matched_embeds]
        self.manager.mine_and_add_objectives(self.known_matched_embeds)
        for obj in self.manager.achieved_objectives:
            self.assertIn(obj.tobytes(), input_set)

    # --- retrieve_last_n_objectives ---

    def test_retrieve_returns_empty_before_mining(self):
        self.assertEqual(self.manager.retrieve_last_n_objectives(5), [])

    def test_retrieve_zero_returns_empty(self):
        self.manager.mine_and_add_objectives(self.known_matched_embeds)
        self.assertEqual(self.manager.retrieve_last_n_objectives(0), [])

    def test_retrieve_returns_exactly_n_when_enough_achieved(self):
        self.manager.mine_and_add_objectives(self.known_matched_embeds)
        total = len(self.manager.achieved_objectives)
        if total < 2:
            self.skipTest("fewer than 2 objectives achieved; cannot test n=2 slice")
        result = self.manager.retrieve_last_n_objectives(2)
        self.assertEqual(len(result), 2)

    def test_retrieve_returns_all_when_n_exceeds_count(self):
        self.manager.mine_and_add_objectives(self.known_matched_embeds)
        total = len(self.manager.achieved_objectives)
        result = self.manager.retrieve_last_n_objectives(total + 100)
        self.assertEqual(len(result), total)

    def test_retrieve_order_matches_insertion_order(self):
        import numpy as np
        self.manager.mine_and_add_objectives(self.known_matched_embeds)
        total = len(self.manager.achieved_objectives)
        result = self.manager.retrieve_last_n_objectives(total)
        for i, obj in enumerate(result):
            np.testing.assert_array_equal(obj, self.manager.achieved_objectives[i])


class TestGetObjectiveDatasetJson(unittest.TestCase):

    def test_returns_unique_video_paths(self):
        result = get_objective_dataset_json([JSON_FILE])
        paths = [r["video_path"] for r in result]
        self.assertEqual(len(paths), len(set(paths)), "video_path entries should be deduplicated")

    def test_each_entry_has_video_path_key(self):
        result = get_objective_dataset_json([JSON_FILE])
        for entry in result:
            self.assertIn("video_path", entry)
            self.assertEqual(list(entry.keys()), ["video_path"], "entry should only have video_path key")

    def test_deduplication_across_multiple_files(self):
        """Same file listed twice should not duplicate entries."""
        result_once = get_objective_dataset_json([JSON_FILE])
        result_twice = get_objective_dataset_json([JSON_FILE, JSON_FILE])
        self.assertEqual(len(result_once), len(result_twice))


class TestLoadObjectiveDataset(unittest.TestCase):

    def setUp(self):
        self.videos_json = get_objective_dataset_json([JSON_FILE])

    def test_embed_and_metadata_lengths_match(self):
        embeds, metadata = load_objective_dataset(self.videos_json, emb_dir=DINOV2_DIR)
        self.assertEqual(embeds.shape[0], len(metadata))

    def test_embed_dim_is_768(self):
        embeds, _ = load_objective_dataset(self.videos_json, emb_dir=DINOV2_DIR)
        self.assertEqual(embeds.shape[1], 768)

    def test_metadata_entries_have_expected_keys(self):
        _, metadata = load_objective_dataset(self.videos_json, emb_dir=DINOV2_DIR)
        for entry in metadata[:10]:
            self.assertIn("video_path", entry)
            self.assertIn("sampled_frame_index", entry)

    def test_metadata_video_paths_match_input(self):
        _, metadata = load_objective_dataset(self.videos_json, emb_dir=DINOV2_DIR)
        input_paths = {item["video_path"] for item in self.videos_json}
        meta_paths = {m["video_path"] for m in metadata}
        self.assertEqual(input_paths, meta_paths)


class TestFilterClusters(unittest.TestCase):

    def test_cluster_with_many_unique_videos_kept(self):
        # 35 unique videos in cluster 0
        meta = [{"video_path": f"vid_{i}.mp4"} for i in range(35)]
        labels = [0] * 35
        result = filter_clusters(labels, meta, min_unique_videos_per_cluster=30)
        self.assertTrue(all(l == 0 for l in result))

    def test_cluster_with_few_unique_videos_becomes_noise(self):
        # only 2 unique videos → should be filtered
        meta = [{"video_path": f"vid_{i % 2}.mp4"} for i in range(20)]
        labels = [0] * 20
        result = filter_clusters(labels, meta, min_unique_videos_per_cluster=30)
        self.assertTrue(all(l == -1 for l in result))

    def test_noise_points_stay_noise(self):
        meta = [{"video_path": f"vid_{i}.mp4"} for i in range(10)]
        labels = [-1] * 10
        result = filter_clusters(labels, meta, min_unique_videos_per_cluster=0)
        self.assertTrue(all(l == -1 for l in result))

    def test_mixed_good_and_bad_clusters(self):
        good_meta = [{"video_path": f"g_{i}.mp4"} for i in range(35)]
        bad_meta  = [{"video_path": f"b_{i % 2}.mp4"} for i in range(10)]
        meta = good_meta + bad_meta
        labels = [0] * 35 + [1] * 10
        result = filter_clusters(labels, meta, min_unique_videos_per_cluster=30)
        self.assertTrue(all(l == 0 for l in result[:35]))
        self.assertTrue(all(l == -1 for l in result[35:]))


class TestObjectivesLookup(unittest.TestCase):
    """Tests ObjectivesLookup using real metadata from disk."""

    def setUp(self):
        import torch
        import numpy as np
        videos_json = get_objective_dataset_json([JSON_FILE])
        self.per_frame_embed, self.per_frame_metadata = load_objective_dataset(
            videos_json[:3], emb_dir=DINOV2_DIR  # 3 videos is enough
        )
        # Attach a fake cluster_embed to a few frames so lookup returns tensors
        for i, meta in enumerate(self.per_frame_metadata):
            if i % 50 == 0:
                meta["cluster_embed"] = self.per_frame_embed[i].tolist()

        self.lookup = ObjectivesLookup(self.per_frame_metadata)

    def test_lookup_before_first_frame_returns_empty(self):
        # frame index 0 should return [] (nothing completed yet)
        video_path = self.per_frame_metadata[0]["video_path"]
        result = self.lookup.lookup(video_path, 0)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    def test_lookup_at_known_video_returns_list(self):
        video_path = self.per_frame_metadata[0]["video_path"]
        last_frame = max(
            m["sampled_frame_index"]
            for m in self.per_frame_metadata
            if m["video_path"] == video_path
        )
        result = self.lookup.lookup(video_path, last_frame + 1)
        self.assertIsInstance(result, list)

    def test_objectives_are_tensors(self):
        import torch
        video_path = self.per_frame_metadata[0]["video_path"]
        last_frame = max(
            m["sampled_frame_index"]
            for m in self.per_frame_metadata
            if m["video_path"] == video_path
        )
        result = self.lookup.lookup(video_path, last_frame + 1)
        for obj in result:
            self.assertIsInstance(obj, torch.Tensor)

    def test_earlier_lookup_has_fewer_objectives(self):
        video_path = self.per_frame_metadata[0]["video_path"]
        frames = sorted(
            m["sampled_frame_index"]
            for m in self.per_frame_metadata
            if m["video_path"] == video_path
        )
        mid = frames[len(frames) // 2]
        last = frames[-1]
        result_mid  = self.lookup.lookup(video_path, mid)
        result_last = self.lookup.lookup(video_path, last + 1)
        self.assertLessEqual(len(result_mid), len(result_last))


class TestMineObjectivesStats(unittest.TestCase):
    """Cluster statistics reported from mine_objectives output."""

    @classmethod
    def setUpClass(cls):
        import time
        from models.train.train_agent import create_clusters, filter_clusters
        videos_json = get_objective_dataset_json(JSON_FILES)
        cls.per_frame_embed, cls.per_frame_metadata = load_objective_dataset(
            videos_json, emb_dir=DINOV2_DIR
        )
        t0 = time.time()
        cls.clusterer = create_clusters(cls.per_frame_embed)
        print(f"\n[HDBSCAN fit] {time.time() - t0:.2f}s  ({len(cls.per_frame_embed)} frames)")
        t1 = time.time()
        cls.filtered_labels = filter_clusters(cls.clusterer.labels_, cls.per_frame_metadata)
        print(f"[filter_clusters] {time.time() - t1:.3f}s")
        cls.valid_cluster_ids = set(l for l in cls.filtered_labels if l != -1)

    def test_report_cluster_stats(self):
        import numpy as np

        labels = self.filtered_labels
        meta   = self.per_frame_metadata

        # Total unique clusters (exclude noise=-1)
        unique_clusters = sorted(set(l for l in labels if l != -1))
        n_clusters = len(unique_clusters)

        # Clusters per video: count distinct cluster IDs per video_path
        clusters_per_video: dict[str, set] = {}
        for label, m in zip(labels, meta):
            vp = m["video_path"]
            if vp not in clusters_per_video:
                clusters_per_video[vp] = set()
            if label != -1:
                clusters_per_video[vp].add(label)

        counts = np.array([len(s) for s in clusters_per_video.values()])
        percentiles = [0, 20, 40, 50, 70, 90, 100]
        pct_values  = np.percentile(counts, percentiles).astype(int)

        print("\n=== Cluster Statistics ===")
        print(f"Total unique clusters : {n_clusters}")
        print(f"Videos               : {len(counts)}")
        print(f"Frames               : {len(labels)}")
        print(f"Noise frames         : {sum(1 for l in labels if l == -1)} "
              f"({100*sum(1 for l in labels if l == -1)/len(labels):.1f}%)")
        print("\nClusters per video (percentiles):")
        for p, v in zip(percentiles, pct_values):
            bar = "#" * int(v)
            print(f"  p{p:>3} : {v:>4}  {bar}")

        self.assertGreater(n_clusters, 0, "Expected at least one cluster")


class TestMineObjectives(unittest.TestCase):
    """End-to-end test of mine_objectives with real data."""

    @classmethod
    def setUpClass(cls):
        cls.result, cls.manager = mine_objectives(JSON_FILES)
        cls.videos_json = get_objective_dataset_json(JSON_FILES)

    def test_returns_objectives_lookup_instance(self):
        self.assertIsInstance(self.result, ObjectivesLookup)

    def test_returns_agent_objective_manager_instance(self):
        self.assertIsInstance(self.manager, AgentObjectiveManager)

    def test_lookup_dict_is_not_empty(self):
        self.assertGreater(len(self.result.lookup_dict), 0)

    def test_all_input_videos_are_present_in_lookup(self):
        for item in self.videos_json:
            self.assertIn(item["video_path"], self.result.lookup_dict,
                          f"{item['video_path']} missing from lookup_dict")

    def test_clustered_videos_have_nonempty_objectives(self):
        """Videos with at least one clustered frame should return objectives for later frames."""
        videos_with_objectives = [
            vp for vp, entries in self.result.lookup_dict.items()
            if any(len(e["objective_embeds"]) > 0 for e in entries)
        ]
        self.assertGreater(len(videos_with_objectives), 0,
                           "Expected at least one video to have clustered objectives")


QUERY_VIDEO = (
    ".cache/pokeagent/online/87a656a8-e16f-45c7-ae5b-d5cf7d182b8f"
    "/query_video/query_gpu0_bootstrap0.mp4"
)


class TestQueryVideoInference(unittest.TestCase):
    """Embed a query video at 2-second intervals and report cluster matches."""

    @classmethod
    def setUpClass(cls):
        import time
        import torch
        import cuml
        from transformers import AutoImageProcessor, AutoModel
        from hdbscan_rapids import _embed_video_every_interval, _predict_hdbscan_labels
        from models.util.misc import local_model_map

        cuml.set_global_output_type("numpy")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = local_model_map("facebook/dinov2-base")
        processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
        model = AutoModel.from_pretrained(model_path).eval().to(device)

        t0 = time.time()
        cls.timestamps, cls.query_embeddings = _embed_video_every_interval(
            video_path=QUERY_VIDEO,
            model=model,
            processor=processor,
            interval_seconds=2.0,
            batch_size=64,
            device=device,
        )
        print(f"\n[embed query video] {time.time() - t0:.2f}s  ({len(cls.timestamps)} frames at 2s intervals)")

        clusterer  = TestMineObjectivesStats.clusterer
        valid_ids  = TestMineObjectivesStats.valid_cluster_ids
        t1 = time.time()
        raw_labels = _predict_hdbscan_labels(clusterer, cls.query_embeddings)
        print(f"[approximate_predict] {time.time() - t1:.3f}s")
        cls.raw_labels = raw_labels.tolist()
        cls.matched_ids = [l for l in cls.raw_labels if l != -1 and l in valid_ids]
        cls.unique_matched = sorted(set(cls.matched_ids))

    def test_query_video_cluster_report(self):
        n_frames   = len(self.timestamps)
        n_matched  = sum(1 for l in self.raw_labels if l != -1 and l in TestMineObjectivesStats.valid_cluster_ids)
        n_noise    = sum(1 for l in self.raw_labels if l == -1)
        n_clusters = len(self.unique_matched)

        print("\n=== Query Video Cluster Inference ===")
        print(f"Video                : {QUERY_VIDEO}")
        print(f"Frames (2s interval) : {n_frames}")
        print(f"Noise frames         : {n_noise} ({100*n_noise/max(n_frames,1):.1f}%)")
        print(f"Matched frames       : {n_matched} ({100*n_matched/max(n_frames,1):.1f}%)")
        print(f"Unique clusters hit  : {n_clusters} / {len(TestMineObjectivesStats.valid_cluster_ids)}")
        print(f"Cluster IDs          : {self.unique_matched}")

        self.assertGreater(n_frames, 0, "Query video produced no frames")

    def test_predicted_labels_length_matches_frames(self):
        self.assertEqual(len(self.raw_labels), len(self.timestamps))

    def test_at_least_one_cluster_matched(self):
        self.assertGreater(
            len(self.unique_matched), 0,
            "Query video matched no valid clusters — check embeddings or cluster params"
        )

    def test_save_cluster_match_images(self):
        """For each matched cluster ID, save the query frame that hit it into tmp/query_cluster_matches/cluster_<id>/."""
        import numpy as np
        from pathlib import Path
        from torchcodec.decoders import VideoDecoder
        from PIL import Image

        valid_ids = TestMineObjectivesStats.valid_cluster_ids
        out_root = Path("tmp/query_cluster_matches")
        out_root.mkdir(parents=True, exist_ok=True)

        decoder = VideoDecoder(QUERY_VIDEO)

        for i, (ts, label) in enumerate(zip(self.timestamps, self.raw_labels)):
            label = int(label)
            if label == -1 or label not in valid_ids:
                continue
            cluster_dir = out_root / f"cluster_{label}"
            cluster_dir.mkdir(exist_ok=True)
            frame_tensor = decoder.get_frames_played_at(seconds=[ts]).data[0]
            frame_np = frame_tensor.cpu().numpy().transpose(1, 2, 0)
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
            img = Image.fromarray(frame_np, mode="RGB")
            img.save(cluster_dir / f"frame_{i:04d}_t{ts:.1f}s.png")

        saved_dirs = sorted(out_root.iterdir())
        print(f"\n[save_cluster_match_images] {len(saved_dirs)} cluster dirs written to {out_root}")
        self.assertTrue(out_root.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
