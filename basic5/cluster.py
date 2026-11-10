"""Bounded spectral clustering of ReDimNet embeddings (not a new neural model)."""
import numpy as np
from .common import require


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] > 0 and np.isfinite(x).all(), "Invalid embedding matrix")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    require((norms > 1e-10).all(), "Zero speaker embedding")
    return x / norms


def cluster_embeddings(features, cfg):
    from scipy.linalg import eigh
    from scipy.sparse.csgraph import connected_components
    from sklearn.cluster import KMeans
    x = normalize(features)
    n = len(x)
    require(n > 0, "No speaker windows")
    sample_indices = np.unique(np.linspace(0, n-1, min(n, cfg["max_cluster_windows"]), dtype=int))
    fit = x[sample_indices]
    cosine = fit @ fit.T
    manual = cfg["num_speakers"]
    regularized = False
    require(manual is None or manual <= len(fit), "More requested speakers than usable windows")
    if len(fit) == 1 or manual == 1:
        k, labels, reason = 1, np.zeros(len(fit), dtype=int), "single_window_or_explicit_one"
    elif manual is None and float(np.median(cosine[np.triu_indices(len(fit), 1)])) >= cfg["single_speaker_similarity"]:
        k, labels, reason = 1, np.zeros(len(fit), dtype=int), "single_speaker_similarity_heuristic"
    else:
        # Positive cosine graph; normalized graph Laplacian; small eigenvectors.
        affinity = np.maximum(cosine, 0.0)
        np.fill_diagonal(affinity, 0.0)
        components = connected_components(affinity > 1e-10, directed=False, return_labels=False)
        if components > 1:
            # Weak edges keep the first eigenvector nonzero on isolated vertices.
            # Component count remains an explicit heuristic, not a quality claim.
            affinity += 1e-6
            np.fill_diagonal(affinity, 0.0)
            regularized = True
        degree = np.maximum(affinity.sum(axis=1), 1e-12)
        lap = np.eye(len(fit)) - affinity / np.sqrt(degree[:, None]*degree[None, :])
        max_k = min(cfg["max_speakers"], len(fit))
        values, vectors = eigh(lap, subset_by_index=[0, min(max_k, len(fit)-1)])
        if manual is not None:
            k, reason = manual, "explicit_speaker_count"
        else:
            gaps = np.diff(values[:max_k+1])
            k = int(np.argmax(gaps)) + 1 if len(gaps) else 1
            k = min(max_k, max(k, components))
            reason = "normalized_laplacian_eigengap_heuristic"
        spectral = normalize(vectors[:, :k])
        labels = KMeans(n_clusters=k, random_state=cfg["seed"], n_init=10).fit_predict(spectral)
    label_map = {label: i for i, label in enumerate(sorted(set(labels)))}
    labels = np.array([label_map[label] for label in labels])
    centers, cancelled_centers = [], 0
    for label in sorted(set(labels)):
        members = fit[labels == label]
        center = members.mean(axis=0)
        if np.linalg.norm(center) <= 1e-10:
            # Opposing embeddings may cancel when a low speaker count is forced.
            center = members[0]
            cancelled_centers += 1
        centers.append(center)
    centers = normalize(np.stack(centers))
    assigned = (x @ centers.T).argmax(axis=1)
    assigned[sample_indices] = labels
    # Presentation IDs follow first occurrence; no reference labels enter clustering.
    mapping = {label: f"S{j+1:02d}" for j, label in enumerate(dict.fromkeys(assigned.tolist()))}
    return [mapping[int(label)] for label in assigned], dict(
        estimated_speakers=len(mapping), reason=reason, fit_windows=len(fit), total_windows=n,
        graph_regularized=regularized, cancelled_centers=cancelled_centers,
        subsampled=len(fit)<n, warning="Speaker-count estimation is heuristic; overlaps are not separated.")


def speaker_windows(vad, cfg):
    windows = []
    for region, (a, b) in enumerate(vad):
        if b-a < cfg["min_speaker_window_seconds"]:
            continue
        length, hop = cfg["speaker_window_seconds"], cfg["speaker_hop_seconds"]
        starts, cursor = [a], a+hop
        while cursor+length <= b:
            starts.append(cursor)
            cursor += hop
        tail = max(a, b-length)
        if tail > starts[-1]+1e-8:
            starts.append(tail)
        for start in starts:
            windows.append(dict(start=start, end=min(b,start+length), region=region))
    return windows


def windows_to_turns(windows, labels, vad):
    require(len(windows) == len(labels), "Window/label count differs")
    turns = []
    for region, (a,b) in enumerate(vad):
        items = [(w,label) for w,label in zip(windows,labels) if w["region"] == region]
        if not items:
            continue  # Very short speech remains unassigned, not silently assigned to a neighbor.
        centers = [(w["start"]+w["end"])/2 for w,_ in items]
        bounds = [a]+[(u+v)/2 for u,v in zip(centers, centers[1:])]+[b]
        for i, (_,label) in enumerate(items):
            if turns and turns[-1]["speaker"] == label and abs(turns[-1]["end"]-bounds[i]) < 1e-8:
                turns[-1]["end"] = bounds[i+1]
            else:
                turns.append(dict(start=bounds[i], end=bounds[i+1], speaker=label))
    return turns
