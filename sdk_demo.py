import argparse
import os

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient


DEFAULT_BASE_URL = "http://127.0.0.1:15666"
DEFAULT_API_KEY = "local"


questions = {
    "department": Choice(
        instructions="Which team should handle this request?",
        criteria={
            "billing": "Payment or subscription issues",
            "technical": "Bugs or integration problems",
            "sales": "Pricing or account questions",
        },
    ),
    "is_urgent": Noul(
        instructions="Does the message convey urgency or time sensitivity?",
    ),
    "frustration": Score(
        instructions="How frustrated does the customer appear?",
        criteria=[
            "Calm and neutral",
            "Concerned but civil",
            "Clearly frustrated",
            "Very angry",
        ],
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Typesafe Python SDK demo")
    parser.add_argument(
        "--base-url",
        default=os.getenv("TYPESAFE_BASE_URL", DEFAULT_BASE_URL),
        help=f"Typesafe API base URL (default: TYPESAFE_BASE_URL or {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("TYPESAFE_API_KEY", DEFAULT_API_KEY),
        help=f"Typesafe API key (default: TYPESAFE_API_KEY or {DEFAULT_API_KEY})",
    )
    args = parser.parse_args()

    client = TypeSafeClient(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    with client:
        result = client.system_one(
            state=(
                "I've been trying to connect my Stripe account for three days. "
                "The integration keeps failing and I'm losing sales."
            ),
            questions=questions,
        )

    print(f"choice: {result.choices['department'].choice}")
    print(f"noul: {result.nouls['is_urgent'].noul}")
    print(f"score: {result.scores['frustration'].score}")


if __name__ == "__main__":
    main()
