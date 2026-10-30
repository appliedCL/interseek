"""Extract development data only; pretrained model files are ignored."""
import argparse
from pairloom.base import extract_development_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = extract_development_data(args.archive, args.output)
    print("Verified development CSV:", data)
    print("No model weights or training wheels were read or extracted.")


if __name__ == "__main__":
    main()
