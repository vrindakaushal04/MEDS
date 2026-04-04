from services.text_emotion import detect

tests = [
    "I am so sad and depressed, cant stop crying",
    "mujhe bahut gussa aa raha hai sab pe",
    "im fine everything is okay dont worry",
    "I am really anxious and scared about everything going on",
    "This is amazing i am so happy today!",
    "I am tired and exhausted, cant sleep at all",
    "please help me i cant take this anymore",
    "I hate this so much, makes me furious",
    "main bilkul theek hoon, koi tension nahi",
    "I feel nervous and worried about my future",
    "feeling numb and empty inside",
    "having a great day, life is wonderful",
]

for t in tests:
    r = detect(t)
    print(f"  [{r['label']:22}] conf={r['confidence']:.2f} | {t[:55]}")
