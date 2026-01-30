"""Small script to verify package-style imports work: run `python -m scripts.check_package_imports`"""

from models import BlackBoxLLM
from wrappers import NoOpWrapper, WrapperDecision


def main():
    m = BlackBoxLLM()
    w = NoOpWrapper()
    out = m.generate("test")
    decision, payload = w.decide("hi", out, [])
    print("model output:", out)
    print("wrapper decision:", decision)


if __name__ == "__main__":
    main()
