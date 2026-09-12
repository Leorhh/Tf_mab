from src.data.candidate_generator import CandidateGenerator


def main():
    generator = CandidateGenerator(num_items=159729, num_candidates=100, seed=42)
    history = [1, 5, 20, 100, 500]
    target = 1000
    candidates, target_idx = generator.generate(target_item=target, history_items=history)

    print("=" * 60)
    print("Candidate Generator Test")
    print("=" * 60)
    print(f"Number of candidates: {len(candidates)}")
    print(f"Target item:          {target}")
    print(f"Target index:         {target_idx}")
    print(f"Target in candidates: {target in candidates}")
    print()
    print("First 20 candidates:")
    print(candidates[:20])
    print()
    print("Target in history:")
    print(target in history)
    print()
    print("Unique candidates:")
    print(len(set(candidates)) == len(candidates))
    print()
    print("History overlap:")
    print(any(item in history for item in candidates))


if __name__ == "__main__":
    main()
