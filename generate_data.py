"""
generate_data.py
----------------
Generates a synthetic but realistic dataset of 1,000 Indian telemarketing
call openings (first 3 lines of the transcript) and saves it to calls.csv.

Each transcript has the structure:
    Agent:    <pitch for a product>
    Customer: <reaction - positive / curious / objection / rejection>
    Agent:    <agent follow-up that responds to the customer>

The label `converted` is drawn from a Bernoulli distribution whose
probability depends on the customer's intent, the call time and the
customer segment.  Because the label is *sampled* (not deterministic),
the data contains realistic noise and no model can reach 100% accuracy.
"""

import random

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 1000
OUTPUT_FILE = "calls.csv"

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
AGENT_NAMES = ["Rahul", "Priya", "Ankit", "Sneha", "Vikram", "Pooja", "Arjun",
               "Neha", "Karan", "Divya", "Rohit", "Meera"]
CUSTOMER_TITLES = ["Sharma ji", "Mr. Verma", "Ms. Iyer", "Mr. Reddy", "Mrs. Gupta",
                   "Mr. Patel", "Ms. Nair", "Mr. Singh", "Mrs. Joshi", "Mr. Khan",
                   "sir", "ma'am"]
COMPANIES = ["Bharat FinServ", "Sahyog Bank", "Trust Capital", "Unity Finance",
             "Suraksha Life", "Navyug Bank"]

PRODUCTS = {
    "credit_card": {
        "pitch": [
            "Good morning {cust}, I am {agent} calling from {co}. You are pre-approved for our Platinum credit card with lifetime free membership.",
            "Hello {cust}, this is {agent} from {co}. We have a special cashback credit card offer selected only for you.",
            "Namaste {cust}, {agent} here from {co}. Based on your profile you are eligible for a premium travel credit card with free lounge access.",
            "Hi {cust}, I am {agent} from {co}. We are offering a rewards credit card with 5 percent cashback on online shopping.",
        ],
        "questions": [
            "Is there an annual fee for this card?",
            "What is the credit limit I will get?",
            "Do I get cashback on Amazon and Flipkart?",
            "Is there any joining fee or hidden charges?",
            "How many reward points per 100 rupees?",
        ],
        "answers": [
            "No sir, the joining fee and annual fee are both waived for you.",
            "Your limit will be around 2 lakh and you get 5 percent cashback.",
            "There are zero hidden charges and the card is lifetime free.",
        ],
    },
    "personal_loan": {
        "pitch": [
            "Hello {cust}, I am {agent} from {co}. You have a pre-approved personal loan of up to 5 lakh at a special interest rate.",
            "Good afternoon {cust}, {agent} calling from {co}. We can disburse an instant personal loan within 24 hours with minimal documents.",
            "Namaste {cust}, this is {agent} from {co}. We have a top-up loan offer for you with no processing fee this month.",
            "Hi {cust}, I am {agent} from {co}. You are eligible for a personal loan with flexible EMI options.",
        ],
        "questions": [
            "What is the interest rate?",
            "How fast is the disbursal?",
            "Is there a processing fee?",
            "What documents are required?",
            "Can I choose the EMI tenure?",
        ],
        "answers": [
            "The rate starts at 10.5 percent and the money comes in 24 hours.",
            "Only PAN, Aadhaar and three months salary slip are needed.",
            "Processing fee is zero this month and you can pick 12 to 60 months EMI.",
        ],
    },
    "health_insurance": {
        "pitch": [
            "Good morning {cust}, I am {agent} from {co}. We have a family health insurance plan with 10 lakh cover and cashless hospitals.",
            "Hello {cust}, {agent} from {co}. I am calling about a health insurance policy that also covers your parents.",
            "Namaste {cust}, this is {agent} calling from {co}. Our health plan gives tax benefit under section 80D.",
        ],
        "questions": [
            "What is the premium per year?",
            "Does it cover my parents also?",
            "Are pre-existing diseases covered?",
            "Which hospitals are in the cashless network?",
        ],
        "answers": [
            "The premium is just 950 rupees per month for the whole family.",
            "Yes, parents are covered and pre-existing diseases are covered after 2 years.",
            "We have more than 8000 cashless hospitals across India.",
        ],
    },
    "term_insurance": {
        "pitch": [
            "Hello {cust}, I am {agent} from {co}. We have a term insurance plan of 1 crore cover at a very low premium.",
            "Good evening {cust}, {agent} calling from {co}. Secure your family's future with our term life plan.",
            "Hi {cust}, this is {agent} from {co}. Our term plan also gives tax saving under section 80C.",
        ],
        "questions": [
            "What will be the monthly premium?",
            "Is there a tax benefit?",
            "What is the claim settlement ratio?",
            "Is medical test required?",
        ],
        "answers": [
            "For your age the premium is around 800 rupees per month.",
            "Yes, you save tax under 80C and the claim settlement ratio is 99 percent.",
            "No medical test is needed up to 50 lakh cover.",
        ],
    },
    "home_loan": {
        "pitch": [
            "Good afternoon {cust}, I am {agent} from {co}. We are offering home loan balance transfer at a lower interest rate.",
            "Hello {cust}, {agent} from {co}. You can save on EMI by moving your home loan to us.",
        ],
        "questions": [
            "What is your home loan interest rate?",
            "How much will my EMI reduce?",
            "Are there any foreclosure charges?",
        ],
        "answers": [
            "Our rate is 8.4 percent so your EMI will reduce by around 3000 rupees.",
            "There are no foreclosure charges and the transfer is free this month.",
        ],
    },
}

# Customer reactions by intent.  `p` is the base probability of conversion.
INTENTS = {
    "positive": {
        "weight": 0.20, "p": 0.96,
        "customer": [
            "Yes, I am interested, please tell me more.",
            "Haan ji, actually I was looking for this only.",
            "Sounds good, how do I apply?",
            "Okay, that is great, please go ahead.",
            "Yes sure, I need this, what is the process?",
            "Achha, that sounds useful. I would like to apply.",
            "Perfect timing, I was planning to get one.",
        ],
        "agent": [
            "Wonderful {cust}, I will just verify a few details and complete the application.",
            "Great, I will share the link and you can complete it in 5 minutes.",
            "Thank you {cust}, let me start your application right now.",
        ],
    },
    "curious": {
        "weight": 0.22, "p": 0.89,
        "customer": ["__QUESTION__"],
        "agent": ["__ANSWER__"],
    },
    "curious_hesitant": {
        "weight": 0.06, "p": 0.25,
        "customer": [
            "Hmm, okay, but __QUESTION_LOWER__",
            "I am not sure. __QUESTION__",
            "Let me think. __QUESTION__",
        ],
        "agent": ["__ANSWER__"],
    },
    "busy": {
        "weight": 0.13, "p": 0.07,
        "customer": [
            "I am in a meeting right now, call me later.",
            "Sorry, I am driving, can you call after some time?",
            "I am busy right now, please call tomorrow.",
            "Abhi time nahi hai, I am in office.",
        ],
        "agent": [
            "Sure {cust}, what time should I call you back?",
            "No problem, I will call you in the evening.",
            "Okay, I will schedule a callback for you.",
        ],
    },
    "whatsapp": {
        "weight": 0.12, "p": 0.15,
        "customer": [
            "Just send the details on WhatsApp, I will check.",
            "Send me the brochure on WhatsApp.",
            "Can you email or WhatsApp me the details?",
            "Message me the details, I will see later.",
        ],
        "agent": [
            "Sure {cust}, I am sending the details on WhatsApp right now.",
            "Okay, I will WhatsApp the brochure and follow up tomorrow.",
        ],
    },
    "price_objection": {
        "weight": 0.10, "p": 0.05,
        "customer": [
            "I already have a card from another bank, I don't need one more.",
            "Your interest rate is too high, other banks are cheaper.",
            "The premium is too expensive for me.",
            "I already have a policy, not required.",
            "Last time there were hidden charges, I don't trust these offers.",
        ],
        "agent": [
            "I understand {cust}, but our offer is better than most banks.",
            "Sir, we can match the rate if you share your current details.",
            "I understand, let me tell you about the extra benefits.",
        ],
    },
    "not_interested": {
        "weight": 0.15, "p": 0.02,
        "customer": [
            "Not interested, please don't call again.",
            "No, I am not interested.",
            "Please remove my number from your list.",
            "Stop calling me, I have told you many times.",
            "Nahi chahiye, thank you.",
            "This is spam, I am not interested at all.",
        ],
        "agent": [
            "Sorry for the disturbance {cust}, have a nice day.",
            "Okay {cust}, I will update your preference.",
            "I understand, thank you for your time.",
        ],
    },
}

CALL_TIMES = ["Morning", "Afternoon", "Evening"]
CALL_TIME_WEIGHTS = [0.35, 0.35, 0.30]
CALL_TIME_EFFECT = {"Morning": -0.03, "Afternoon": -0.05, "Evening": 0.07}

SEGMENTS = ["Salaried", "Self-Employed", "Student"]
SEGMENT_WEIGHTS = [0.55, 0.30, 0.15]


def segment_effect(segment: str, product: str) -> float:
    """Business-logic adjustment of conversion probability."""
    if segment == "Salaried":
        return 0.06
    if segment == "Self-Employed":
        return 0.05 if product in ("personal_loan", "home_loan") else 0.0
    # Students rarely qualify for loans / premium products
    if product in ("personal_loan", "home_loan", "term_insurance"):
        return -0.20
    return -0.08


def make_row() -> dict:
    product = random.choice(list(PRODUCTS.keys()))
    prod = PRODUCTS[product]
    intent_name = random.choices(list(INTENTS.keys()),
                                 weights=[v["weight"] for v in INTENTS.values()])[0]
    intent = INTENTS[intent_name]

    fmt = {"agent": random.choice(AGENT_NAMES),
           "cust": random.choice(CUSTOMER_TITLES),
           "co": random.choice(COMPANIES)}

    line1 = random.choice(prod["pitch"]).format(**fmt)

    question = random.choice(prod["questions"])
    line2 = random.choice(intent["customer"])
    line2 = (line2.replace("__QUESTION_LOWER__", question[0].lower() + question[1:])
                  .replace("__QUESTION__", question))

    line3 = random.choice(intent["agent"])
    line3 = line3.replace("__ANSWER__", random.choice(prod["answers"])).format(**fmt)

    call_time = random.choices(CALL_TIMES, weights=CALL_TIME_WEIGHTS)[0]
    segment = random.choices(SEGMENTS, weights=SEGMENT_WEIGHTS)[0]

    p = intent["p"] + CALL_TIME_EFFECT[call_time] + segment_effect(segment, product)
    p = float(np.clip(p, 0.01, 0.98))
    converted = int(np.random.rand() < p)

    transcript = f"Agent: {line1}\nCustomer: {line2}\nAgent: {line3}"
    return {
        "transcript_opening": transcript,
        "agent_lines": f"{line1} {line3}",      # caller (agent) speech only
        "customer_response": line2,             # client speech only - used by the model
        "call_time": call_time,
        "customer_segment": segment,
        "product": product,
        "converted": converted,
    }


def main() -> None:
    df = pd.DataFrame([make_row() for _ in range(N_ROWS)])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
    print(f"Conversion rate: {df['converted'].mean():.2%}")
    print("\nConversion rate by call time:")
    print(df.groupby("call_time")["converted"].mean().round(3).to_string())
    print("\nConversion rate by segment:")
    print(df.groupby("customer_segment")["converted"].mean().round(3).to_string())
    print("\nSample transcript:\n" + df.loc[0, "transcript_opening"])


if __name__ == "__main__":
    main()
