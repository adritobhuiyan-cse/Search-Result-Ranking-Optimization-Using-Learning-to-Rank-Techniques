"""
STEP 1 (OFFLINE FALLBACK) — Generates realistic synthetic MS MARCO-style data
Use this ONLY if Hugging Face download fails (firewall, no internet, etc.)

Generates 5,000 queries x ~8 passages each = ~40,000 rows
with realistic relevance distribution (~1 relevant per query).

Run: python step1_offline_fallback.py
"""

import random
import numpy as np
import pandas as pd

OUT_FILE   = "raw_msmarco.csv"
N_QUERIES  = 5000
PASSAGES_PER_QUERY = 8
random.seed(42)
np.random.seed(42)

# Realistic query templates
QUERY_TEMPLATES = [
    "what is {topic}", "how does {topic} work", "why is {topic} important",
    "what causes {topic}", "how to treat {topic}", "history of {topic}",
    "what are the effects of {topic}", "how to prevent {topic}",
    "definition of {topic}", "types of {topic}", "symptoms of {topic}",
    "best way to {action}", "how long does {topic} last",
    "is {topic} dangerous", "what does {topic} mean",
]

TOPICS = [
    "diabetes", "photosynthesis", "gravity", "inflation", "machine learning",
    "climate change", "antibiotics", "black holes", "democracy", "evolution",
    "cancer", "alzheimers disease", "solar energy", "artificial intelligence",
    "depression", "vaccines", "earthquakes", "the french revolution",
    "the water cycle", "dna replication", "mitosis", "blockchain",
    "quantum computing", "the immune system", "plate tectonics",
    "cognitive behavioral therapy", "the stock market", "interest rates",
    "sleep deprivation", "the human brain", "the industrial revolution",
    "protein synthesis", "osmosis", "natural selection", "the big bang",
    "nuclear fusion", "the ozone layer", "acid rain", "the food chain",
    "the roman empire", "the magna carta", "the civil war", "world war 2",
    "the cold war", "the renaissance", "the enlightenment", "globalization",
    "poverty", "immigration", "gun control", "the death penalty",
]

ACTIONS = [
    "lose weight", "improve memory", "sleep better", "reduce stress",
    "learn a language", "save money", "stay motivated", "build muscle",
    "improve posture", "quit smoking", "eat healthier", "focus better",
]

# Passage sentence pools
INTRO_SENTENCES = [
    "{topic} is a complex phenomenon that affects millions of people worldwide.",
    "{topic} refers to the process by which organisms adapt to their environment.",
    "Scientists have studied {topic} for decades without reaching a consensus.",
    "{topic} was first described by researchers in the early twentieth century.",
    "The study of {topic} has led to significant medical breakthroughs.",
    "{topic} is commonly misunderstood by the general public.",
    "There are several theories that attempt to explain {topic}.",
    "{topic} plays a crucial role in the functioning of modern society.",
    "Understanding {topic} requires a background in basic science.",
    "{topic} has been linked to a number of health outcomes in recent studies.",
]

DETAIL_SENTENCES = [
    "Research shows that early intervention can significantly improve outcomes.",
    "The mechanism involves a complex series of chemical reactions.",
    "Studies have identified several risk factors associated with this condition.",
    "Treatment options vary depending on the severity and duration of symptoms.",
    "Prevention strategies focus on lifestyle modification and education.",
    "The process requires specific environmental conditions to occur.",
    "Multiple factors contribute to the development of this phenomenon.",
    "Scientists are still investigating the underlying causes.",
    "The effects can be observed at both the cellular and systemic levels.",
    "Clinical trials have demonstrated significant efficacy in controlled settings.",
    "The long-term consequences remain an active area of investigation.",
    "Experts recommend consulting a qualified professional for personalized advice.",
]

def make_query():
    template = random.choice(QUERY_TEMPLATES)
    if "{action}" in template:
        return template.replace("{action}", random.choice(ACTIONS))
    return template.replace("{topic}", random.choice(TOPICS))

def make_passage(query_topic, is_relevant):
    """Generate a passage that is relevant or not to the topic."""
    if is_relevant:
        topic = query_topic
    else:
        # Pick a different topic
        other_topics = [t for t in TOPICS if t != query_topic]
        topic = random.choice(other_topics)

    intro   = random.choice(INTRO_SENTENCES).replace("{topic}", topic)
    details = " ".join(random.sample(DETAIL_SENTENCES, k=random.randint(2, 4)))
    return f"{intro} {details}"


def main():
    print("=" * 60)
    print("STEP 1 (OFFLINE FALLBACK) — Generating synthetic MS MARCO data")
    print("=" * 60)
    print(f"\nGenerating {N_QUERIES:,} queries x {PASSAGES_PER_QUERY} passages...")

    records = []
    for qid in range(N_QUERIES):
        query = make_query()
        # Extract the topic from the query for passage generation
        topic = query.split()[-1] if query.split() else "health"

        # One randomly-placed relevant passage, rest irrelevant
        relevant_idx = random.randint(0, PASSAGES_PER_QUERY - 1)

        for offset in range(PASSAGES_PER_QUERY):
            is_relevant = (offset == relevant_idx)
            passage     = make_passage(topic, is_relevant)
            records.append({
                'qid':       qid,
                'pid':       qid * 100 + offset,
                'query':     query,
                'passage':   passage,
                'relevance': int(is_relevant),
            })

    df = pd.DataFrame(records)

    print(f"\nGenerated:")
    print(f"  Rows:     {len(df):,}")
    print(f"  Queries:  {df['qid'].nunique():,}")
    rel = df['relevance'].value_counts()
    for v in sorted(rel.index):
        print(f"  relevance={v}: {rel[v]:,} rows ({100*rel[v]/len(df):.1f}%)")

    df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved to {OUT_FILE}")
    print(f"\nNext step: python step2_clean_data.py")


if __name__ == "__main__":
    main()
