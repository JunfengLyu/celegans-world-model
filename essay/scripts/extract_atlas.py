import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

WRL_URL = "https://raw.githubusercontent.com/openworm/Blender2NeuroML/598081825349ed72de788f005db5b30ff5168ffa/src/Data/Virtual_Worm_March_2011.wrl"
MORPHOLOGY_URL = "https://github.com/openworm/c302/tree/d542a44bf2662f79e0ad23342a959a181a920877/c302/NeuroML2"


def read_meshes(wrl):
    text = wrl.read_text()
    object_starts = list(re.finditer(r"^\tDEF (\S+)\s*$", text, re.M))
    transforms = {
        name: np.fromstring(block, sep=" ").reshape(4, 4)
        for block, name in re.findall(r"MatrixTransform\s*\{\s*matrix\s*([^{}]+)\}\s*USE (\S+)", text)
    }
    mesh = {}
    wanted = re.compile(r"^Cuticle$|^pm\d|^int\d|^rect_|^Gonadal_Sheath_|^vul[A-Z]")
    for entry, following in zip(object_starts, object_starts[1:]):
        name = entry.group(1)
        if name not in transforms or not wanted.match(name):
            continue
        block = text[entry.end():following.start()]
        coordinates = re.search(r"Coordinate3\s*\{\s*point\s*\[([^\]]+)\]", block)
        if coordinates is None:
            continue
        points = np.fromstring(coordinates.group(1).replace(",", " "), sep=" ").reshape(-1, 3)
        points = (np.column_stack((points, np.ones(len(points)))) @ transforms[name])[:, :3] * 100
        polygons = []
        for face_set in re.findall(r"coordIndex\s*\[([^\]]+)\]", block):
            for face in re.split(r"-1,?", face_set):
                ids = np.fromstring(face.replace(",", " "), sep=" ", dtype=int)
                if len(ids) > 2:
                    polygons.append(ids)
        mesh[name] = points, polygons
    return mesh


def edges(mesh):
    points, faces = mesh
    pairs = np.concatenate([np.stack((face, np.roll(face, 1)), axis=-1) for face in faces])
    pairs = np.unique(np.sort(pairs, axis=-1), axis=0)
    return points[pairs]


def slice_points(segs, longitudes):
    location = segs[:, :, 1]
    a, b = location.T
    return [
        (segs[active, 0] + ((coord - a[active]) / (b[active] - a[active]))[:, None]
         * (segs[active, 1] - segs[active, 0]))
        for coord in longitudes
        for active in [((a <= coord) & (b >= coord) | (b <= coord) & (a >= coord)) & (a != b)]
    ]


def simplify(points, tolerance):
    """Douglas–Peucker decimation in three dimensions, preserving each branch."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    difference = end - start
    denom = difference @ difference
    if denom == 0:
        distance = np.linalg.norm(points - start, axis=1)
    else:
        ratio = np.clip((points - start) @ difference / denom, 0, 1)
        distance = np.linalg.norm(points - start - ratio[:, None] * difference, axis=1)
    index = int(np.argmax(distance))
    if distance[index] <= tolerance:
        return points[[0, -1]]
    return np.concatenate((simplify(points[:index + 1], tolerance)[:-1],
                           simplify(points[index:], tolerance)))


def read_morph(path):
    xml = ET.parse(path)
    segments = xml.findall(".//{*}morphology/{*}segment")
    if not segments:
        raise ValueError(f"No morphology segments in {path}")
    nodes = {segment.attrib["id"]: segment for segment in segments}
    children = defaultdict(list)

    def point(element):
        return np.array([float(element.attrib[axis]) for axis in ("x", "y", "z")])

    def distal(item):
        return point(nodes[item].find("{*}distal"))

    def proximal(item):
        node = nodes[item]
        current = node.find("{*}proximal")
        if current is not None:
            return point(current)
        parent = node.find("{*}parent")
        parent_id = parent.attrib["segment"]
        t = float(parent.attrib.get("fractionAlong", 1.0))
        if t == 1:
            return distal(parent_id)
        return (1 - t) * proximal(parent_id) + t * distal(parent_id)

    roots = []
    for id_, segment in nodes.items():
        parent = segment.find("{*}parent")
        if parent is None:
            roots.append(id_)
        else:
            children[parent.attrib["segment"]].append(id_)

    branches = []

    def follow(item, head):
        if not np.allclose(head[-1], proximal(item), atol=1e-3):
            if len(head) > 1:
                branches.append(head)
            head = [proximal(item)]
        tail = distal(item)
        if not np.allclose(head[-1], tail, atol=1e-4):
            head.append(tail)
        successors = children[item]
        if len(successors) == 1:
            follow(successors[0], head)
        else:
            if len(head) > 1:
                branches.append(head)
            for successor in successors:
                follow(successor, [tail])

    for root in roots:
        follow(root, [proximal(root)])
    return distal(roots[0]), [np.array(branch) for branch in branches]


def main(args):
    meshes = read_meshes(args.mesh)
    cut_points, _ = meshes["Cuticle"]
    front, rear = np.min(cut_points[:, 1]), np.max(cut_points[:, 1])
    length = rear - front
    longitudes = np.linspace(front, rear, 180)
    section = slice_points(edges(meshes["Cuticle"]), longitudes)
    low = np.array([np.min(s, axis=0) for s in section])
    high = np.array([np.max(s, axis=0) for s in section])
    mid = (low + high) / 2

    def normalize(points):
        lateral, length_coord, dorsal = points.T
        mean_x = np.interp(length_coord, longitudes, mid[:, 0])
        mean_z = np.interp(length_coord, longitudes, mid[:, 2])
        return np.stack((np.clip((length_coord - front) / length, 0, 1),
                         (dorsal - mean_z) / length, (lateral - mean_x) / length), axis=-1)

    def rounded(points):
        return np.round(points, 4).tolist()

    outline = np.concatenate((np.stack((mid[:, 0], longitudes, high[:, 2]), axis=1),
                              np.stack((mid[:, 0], longitudes, low[:, 2]), axis=1)[::-1]))
    landmarks, landmark_outlines = {}, {}
    for label, prefix in (("pharynx", r"^pm\d"), ("intestine", r"^int\d"),
                          ("gonad", r"^Gonadal_Sheath_"), ("vulva", r"^vul[A-Z]"),
                          ("rectum", r"^rect_")):
        organ_meshes = [mesh for name, mesh in meshes.items() if re.match(prefix, name)]
        combined = np.concatenate([edges(mesh) for mesh in organ_meshes])
        resolution = 65 if label in ("intestine", "gonad") else 32
        xs = np.linspace(combined[:, :, 1].min(), combined[:, :, 1].max(), resolution)
        slices = slice_points(combined, xs)
        pairs = [(s.min(axis=0), s.max(axis=0)) for s in slices if len(s)]
        if not pairs:
            continue
        lows, highs = np.array(pairs).transpose((1, 0, 2))
        centered = (lows + highs) / 2
        landmarks[label] = rounded(normalize(centered))
        highs[:, 0] = lows[:, 0] = centered[:, 0]
        landmark_outlines[label] = rounded(normalize(np.concatenate((highs, lows[::-1]))))

    names = [entry["name"] for entry in sorted(json.loads(args.neurons.read_text())["neurons"],
                                               key=lambda entry: entry["index"])]
    xyz_soma, neurites = [], []
    per_neuron = {}
    for name in names:
        alias = re.sub(r"^([A-Z]+)0+(\d+)$", r"\1\2", name)
        per_neuron[name] = read_morph(args.morph / f"{alias}.cell.nml")
    for soma, branches in per_neuron.values():
        xyz_soma.append(rounded(normalize(soma[None])[0]))
        neurites.append([rounded(normalize(simplify(branch, args.tolerance)).ravel())
                         for branch in branches])

    # Cross-sections remain empirically bounded by the cuticle mesh. Store y/z
    # semi-extents for drawing 3-D elliptical sections in a thin atlas view.
    cross_sections = rounded(np.column_stack(((longitudes - front) / length,
                                              (high[:, 2] - low[:, 2]) / (2 * length),
                                              (high[:, 0] - low[:, 0]) / (2 * length))))
    data = {
        "names": names,
        "soma": xyz_soma,
        "neurites": neurites,
        "outline": rounded(normalize(outline)),
        "crossSections": cross_sections,
        "landmarks": landmarks,
        "landmarkOutlines": landmark_outlines,
        "source": {
            "mesh": WRL_URL,
            "morphologies": MORPHOLOGY_URL,
            "description": "Anatomy reconstructed from the Virtual Worm reference atlas; c302 neurites converted from Virtual Worm. Cuticle and organ contours are sagittal projections. Atlas anatomy is not an optical recording of the simulated individual.",
            "coordinates": "Dimensionless body-length units. x from anterior (0) to posterior (1); y from source-VRML z, z from source-VRML x. Both radial coordinates recentered using cuticle cross-sections before normalization.",
            "lengthMicrometers": round(length, 3),
            "neuriteDecimationMicrometers": args.tolerance,
            "matchedNeurons": len(names),
            "neuronPaths": sum(map(len, neurites)),
        },
    }
    args.out.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print(f"{args.out} • {args.out.stat().st_size:,} B · {len(names)} somata · {sum(map(len, neurites))} neurite paths · {sum(len(l)//3 for neuron in neurites for l in neuron)} path vertices · {length:.1f} µm cuticle")
    print("Soma coordinate ranges:", np.array(xyz_soma).min(0), np.array(xyz_soma).max(0))
    print("Atlas region extent:", {name: [round(pts[0][0], 3),round(pts[-1][0], 3)] for name,pts in landmarks.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a compact registered Virtual Worm anatomy projection")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--morph", type=Path, required=True)
    parser.add_argument("--neurons", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.8)
    main(parser.parse_args())
