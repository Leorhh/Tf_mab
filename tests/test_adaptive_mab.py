import numpy as np
from src.bandit.adaptive_mab import AdaptiveMAB


def main():
    mab = AdaptiveMAB(beta_min=0.1, beta_max=1.0, tau=0.01)
    cases = {
        "Low uncertainty": np.array([0.001, 0.002, 0.003, 0.002]),
        "Medium uncertainty": np.array([0.01, 0.015, 0.02, 0.012]),
        "High uncertainty": np.array([0.03, 0.05, 0.08, 0.06]),
    }

    print("=" * 60)
    print("Adaptive MAB Uncertainty Test")
    print("=" * 60)
    for name, unc in cases.items():
        beta = mab.compute_beta(unc)
        print()
        print(name)
        print(f"Mean uncertainty: {np.mean(unc):.6f}")
        print(f"Adaptive beta:    {beta:.6f}")


if __name__ == "__main__":
    main()
