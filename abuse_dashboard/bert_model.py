from transformers import pipeline
from neo4j import GraphDatabase
import re
import torch  
import pandas as pd
import gc
import warnings
import uuid 
import random
warnings.filterwarnings("ignore")  # Ignore tokenizer warnings

URI = "neo4j+s://d062fece.databases.neo4j.io"
USER = "neo4j"
PASSWORD = "B941VfHLuKnfx5EDR2z4HrZCc-NX6NcACXPk4pHMGrY"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# LABEL 1
always_does_this = {
    "Adrenaline junky": "too intense",
    "Better than you": "You should be in awe of my intelligence and look up to me intellectually and I know better than you do or even when it is about you",
    "Blame-shifting": "He increasingly casts the blame to you for everything he is unhappy about",
    "Boundary-pushing": "Pushes to see you all the time",
    "Chameleon": "He is a chameleon and seems to be whatever you want him to be",
    "Charms with skill":"",
    "Controlling": "highly controlling and do not respect your boundaries",
    "Dependent": "You give more to him than to a newborn but it is never enough",
    "Dysregulated":"",
    "Emotionally manipulative": "They use passive-aggressive behavior to manipulate you",
    "Emotionally unavailable": "He is emotionally unavailable or distant ",
    "Entitled": "If your needs conflict with his you are self-centered",
    "Evasive or deceptive": "Avoids questions about his past or provides inconsistent information",
    "Fear of Abandonment": "Fears rejection above all else so jealous of everyone in your life",
    "Future fakes": "Promises significant change or commitment in the future to placate or control you but never follows through",
    "Good in bed": "He gives you the impression that his sexual interest equals intimate interest",
    "Hot and cold": "He is hot and cold or inconsistent in affection and attention",
    "Intimidating": "He was so deeply wounded that he had no other choice but to intimidate you with his anger",
    "Lack of guilt": "I had it so hard I am not responsible for my actions",
    "Love bombs": "Gives you lots of attention but moves too fast",
    "Pity play": "Needs constant reassurance and does this by invoking your pity",
    "Sadistic": "Sadistic and finds cruelty thrilling"
}

# LABEL 2
LABEL_MAP = {
    "Psychological Manipulation": [
        "gaslighting",
        "emotional_manipulation",
        "love_bombing",
        "guilt_tripping",
        "stonewalling",
        "triangulation"
    ],
    "Control and Power": [
        "controlling_behavior",
        "using_privilege_treating_as_objects",
        "withholding",
        "isolation",
        "puts_himself_on_a_pedestal"
    ],
    "Abusive Dynamics": [
        "moving_the_goalposts",
        "negging",
        "contempt",
        "verbal_abuse",
        "belittling",
        "blaming_and_shaming",
        "criticism"
    ],
    "Defensive/Evasive Behaviors": [
        "defensiveness",
        "dismissive",
        "entitlement",
        "playing_the_victim"
    ]
}

# === Labels ===
always_does_this_labels = list(always_does_this.keys())

ALL_SUBLABELS = [
    sublabel
    for sublist in LABEL_MAP.values()
    for sublabel in sublist
]

def analyze_message(
    text: str,
    top_n_categories: int = 2,
    top_n_behaviors: int = 5,
    top_n_typologies: int = 3,
    top_n_techniques: int = 3,
    score_threshold: float = 0.1,
    behavior_labels=None
):
    print("**Input message:**")
    print(f"> {text}\n")

    # 2) Classify categories
    cat_labels = list(LABEL_MAP.keys())
    res_cat = classifier(
        sequences=text,
        candidate_labels=cat_labels,
        multi_label=True
    )
    cat_preds = list(zip(res_cat["labels"], res_cat["scores"]))[:top_n_categories]
    if not cat_preds:
        print("_No clear category detected._\n")
        return

    print(f"**Top {top_n_categories} categories:**")
    for name, score in cat_preds:
        print(f"- {name}: {score:.2f}")
    print()

    # 3) Classify sublabels per top category
    category_sublabels = {}
    for cat, _ in cat_preds:
        sublabels = LABEL_MAP.get(cat, [])
        res_sub = classifier(
            sequences=text,
            candidate_labels=sublabels,
            multi_label=True
        )
        sub_preds = sorted(zip(res_sub["labels"], res_sub["scores"]), key=lambda x: x[1], reverse=True)
        top_subs_cat = [lbl for lbl, sc in sub_preds if sc >= score_threshold][:top_n_techniques]
        category_sublabels[cat] = top_subs_cat

    top_subs = [lbl for subs in category_sublabels.values() for lbl in subs]
    print(f"**Top {top_n_techniques} techniques per category:**")
    for cat, subs in category_sublabels.items():
        print(f"- {cat}: {', '.join(subs)}")
    print()

    # 4) Classify behaviors
    beh_labels = behavior_labels or always_does_this_labels
    res_beh = classifier(
        sequences=text,
        candidate_labels=beh_labels,
        multi_label=True
    )
    beh_preds = list(zip(res_beh["labels"], res_beh["scores"]))[:top_n_behaviors]
    if not beh_preds:
        print("_No behaviors detected._\n")
        return

    print(f"**Top {top_n_behaviors} behavior scores:**")
    print("| Behavior             | Score |")
    print("|----------------------|-------|")
    for lbl, sc in beh_preds:
        print(f"| {lbl:20} | {sc:5.2f} |")
    print()

    # 5) Fetch typologies & red flags
    score_map = dict(beh_preds)
    labels = list(score_map.keys())
    with driver.session() as session:
        recs = session.run(
            """
            UNWIND $labels AS behavior
            MATCH (p:PlayerTypology)-[:ALWAYS_DOES_THIS]->(:AlwaysDoes {name: behavior})
            OPTIONAL MATCH (p)-[:SHOWS]->(rf:RedFlag)
            RETURN behavior, p.name AS typology, p.description AS summary,
                   COLLECT(DISTINCT rf.name) AS redflags
            """,
            {"labels": labels}
        )
        typ_records = list(recs)

    # 6) Merge & score typologies
    merged = {}
    for r in typ_records:
        typ = r["typology"]
        sc = score_map[r["behavior"]]
        flags = set(r.get("redflags", []))
        if typ not in merged:
            merged[typ] = {
                "summary": r["summary"],
                "total_score": sc,
                "matched_behaviors": [r["behavior"]],
                "redflags": flags
            }
        else:
            e = merged[typ]
            e["matched_behaviors"].append(r["behavior"])
            e["redflags"].update(flags)
            e["total_score"] = sum(score_map[b] for b in e["matched_behaviors"])

    top_types = sorted(
        merged.items(), key=lambda x: x[1]["total_score"], reverse=True
    )[:top_n_typologies]

    # 7) Select red flags across typologies
    counts = [3, 2, 2]
    selected_redflags = []
    for (typ, info), count in zip(top_types, counts):
        available = list(info["redflags"] - set(selected_redflags))
        selected = random.sample(available, min(count, len(available)))
        selected_redflags.extend(selected)

    # 8) Fetch techniques per typology
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:PlayerTypology)-[:USES]->(ab:AbuseTechnique)
            WHERE p.name IN $names
            RETURN p.name AS typology, COLLECT(DISTINCT ab.name) AS techniques
            """,
            {"names": [t for t, _ in top_types]}
        )
        tech_map = {r["typology"]: r["techniques"] for r in result}

    # 9) Match techniques
    all_matches = set()
    for typ, info in top_types:
        allowed = set(tech_map.get(typ, []))
        matches = allowed.intersection(top_subs)
        all_matches.update(matches)

        print(f"\n## {typ} — total score: {info['total_score']:.2f}")
        print("**Matched behaviors:**", ", ".join(info["matched_behaviors"]))
        print("**Summary:**")
        print(f"> {info['summary']}\n")

    print("**Red flags to watch:**")
    for rf in selected_redflags:
        print(f"- {rf}")

    if all_matches:
        print(
            "\n**Based on categories & typologies, watch out for:**\n  - "
            + "\n  - ".join(sorted(all_matches))
        )
    else:
        print("\n_No matching abuse techniques to highlight based on your text._")


    # ✅ Final return for FastAPI
    abuse_categories = [
    {
        "category": cat,
        "techniques": [{"name": t} for t in category_sublabels.get(cat, [])]
    }
    for cat, _ in cat_preds
    ]

    return {
    "input_message": text,
    "typologies": [
        {
            "name": typ,
            "summary": info["summary"]
        }
        for typ, info in top_types
    ],
    "red_flags": selected_redflags,
    "abuse_techniques": sorted(all_matches),
    "abuse_categories": abuse_categories
    }


