"""A small is-a taxonomy with DESCRIPTIVE glosses for training/eval.

Critical design choice: a node's gloss never names its ancestors. If a child's
text contained its parent's name, cosine could ace ancestor retrieval by mere
lexical overlap and there'd be nothing for hyperbolic geometry to prove. So
general concepts get general descriptions that share little vocabulary with
their specific descendants — exactly the asymmetric setting where cosine is
weak and a trained hyperbolic chart wins.

Structure: 4 domains x 3 subtopics x 3 leaves = 52 nodes, depth 0..2.
`build_store()` embeds every gloss and returns a populated MemoryStore whose
parent/child links are the is-a edges used as free training supervision.
"""

from __future__ import annotations

# (gloss, {subtopic_gloss: [leaf_gloss, ...]})
TAXONOMY: dict[str, dict[str, list[str]]] = {
    "a living organism that moves breathes and reacts to its surroundings": {
        "a warm blooded creature that nurses its young with milk and has hair": [
            "a loyal four legged companion that barks and wags its tail",
            "an independent climbing pet that purrs and hunts small prey",
            "an enormous ocean dweller that spouts water and sings deep songs",
        ],
        "a feathered egg laying flyer with a beak and hollow bones": [
            "a sharp eyed soaring hunter that snatches prey with strong talons",
            "a tuxedo colored swimmer of the cold south that cannot fly",
            "a tiny chirping garden visitor that hops and pecks at seeds",
        ],
        "a cold blooded scaly crawler that basks in the warm sun": [
            "a long limbless slithering hunter that swallows its meals whole",
            "a small darting wall climber that can shed and regrow its tail",
            "a slow armored wanderer that hides inside a hard domed shell",
        ],
    },
    "a built machine that carries people or goods from place to place": {
        "a wheeled mover that travels over roads and solid ground": [
            "a four wheeled passenger box steered with a wheel and pedals",
            "a two wheeled pedal powered frame balanced by the rider",
            "a long coupled chain of cars that runs along steel rails",
        ],
        "a craft that rises into the sky and travels through the air": [
            "a winged metal tube with jet engines that cruises far above clouds",
            "a hovering rotor craft that lifts straight up and lands anywhere",
            "an engineless winged craft that rides rising currents in silence",
        ],
        "a vessel that floats and moves across water": [
            "a towering steel hull that ferries cargo across wide oceans",
            "a sealed diving craft that travels hidden beneath the waves",
            "a slim open paddle craft that glides quietly along calm rivers",
        ],
    },
    "an edible substance that nourishes the body and gives energy": {
        "a sweet fleshy plant product grown to be eaten ripe": [
            "a crisp round orchard pick that is red or green and tart sweet",
            "a soft yellow tropical crescent peeled from a thick skin",
            "a small juicy cluster bead eaten fresh or pressed into wine",
        ],
        "an edible plant part eaten as a savory side or main": [
            "a crunchy orange root pulled from the soil and rich in vitamins",
            "a leafy dark green bundle wilted or tossed raw in salads",
            "a starchy underground tuber boiled mashed or fried in oil",
        ],
        "a sweet treat served at the end of a meal as a reward": [
            "a layered baked sponge frosted and sliced for celebrations",
            "a frozen churned dairy scoop served cold in a cone or cup",
            "a small flat sweet baked round that is crisp or chewy",
        ],
    },
    "a crafted device used to produce musical sound and melody": {
        "an instrument sounded by plucking or bowing tensioned strings": [
            "a six stringed fretted body strummed or picked with the fingers",
            "a four stringed wooden voice bowed under the chin in orchestras",
            "a tall triangular frame of many strings plucked with both hands",
        ],
        "an instrument sounded by striking a surface or body": [
            "a stretched skin struck with sticks to keep a steady beat",
            "a row of tuned wooden bars tapped to ring out bright notes",
            "a small jingling hoop shaken or slapped against the palm",
        ],
        "an instrument sounded by blowing air through a channel": [
            "a slim metal pipe held sideways and blown across an open hole",
            "a brass valved horn buzzed into for bold bright fanfares",
            "a black wooden tube with keys and a single vibrating reed",
        ],
    },
}


def build_store(engine_factory=None):
    """Insert the taxonomy into a fresh KMAEngine's store and return the engine.

    Heuristic coordinates are filled in on insert; training overwrites them with
    learned coordinates. We only rely here on embeddings + parent/child edges.
    """
    from kma.engine import KMAEngine

    eng = (engine_factory or KMAEngine)()
    for root_gloss, subs in TAXONOMY.items():
        root = eng.insert(root_gloss, topic_label=root_gloss[:24])
        for sub_gloss, leaves in subs.items():
            sub = eng.insert(sub_gloss, parent_id=root.id, topic_label=sub_gloss[:24])
            for leaf_gloss in leaves:
                eng.insert(leaf_gloss, parent_id=sub.id)
    return eng
