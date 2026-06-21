"""End-to-end demo: insert a small hierarchy, query, draw the disk.

    python demo.py
"""

from __future__ import annotations

from kma.engine import KMAEngine


def main() -> None:
    eng = KMAEngine()

    space = eng.insert("space exploration", topic_label="space")
    mars = eng.insert("missions to the planet mars", parent_id=space.id, topic_label="mars")
    eng.insert("rovers drive across the martian surface collecting rock samples", parent_id=mars.id)
    eng.insert("mars has a thin cold carbon dioxide atmosphere", parent_id=mars.id)

    food = eng.insert("home cooking", topic_label="food")
    bread = eng.insert("baking bread at home", parent_id=food.id, topic_label="bread")
    eng.insert("knead the dough then let the yeast make it rise", parent_id=bread.id)

    print(f"embedder: {eng.embedder.name}   nodes: {len(eng.store)}\n")

    q = "what is the surface of mars like"
    print(f'query: "{q}"\n')
    for h in eng.query(q, k=4):
        print(f"  [{h.via:6}] score={h.score:.3f} cos={h.cosine:.3f} "
              f"d2anchor={h.hyp_to_anchor:.2f}  d{h.node.depth}  {h.node.text[:48]}")

    try:
        path = __import__("kma.viz", fromlist=["plot_disk"]).plot_disk(eng.store)
        print(f"\ndisk plot saved -> {path}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(viz skipped: {e})")


if __name__ == "__main__":
    main()
