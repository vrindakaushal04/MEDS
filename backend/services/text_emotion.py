"""Text emotion: rich keyword cues + VADER sentiment — detects all emotions accurately."""

from __future__ import annotations

import re

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except Exception:
    _vader = None


# ── Positive / Happy ────────────────────────────────────────────────────────
_POS = re.compile(
    r"\b(happy|happiness|glad|joy|joyful|joyous|excited|exciting|great|love|loving|"
    r"wonderful|amazing|awesome|fantastic|brilliant|incredible|beautiful|delightful|"
    r"thankful|grateful|thanks|thank you|good|better|best|perfect|excellent|"
    r"blessed|proud|relieved|calm|peaceful|content|satisfied|cheerful|"
    r"thrilled|elated|ecstatic|euphoric|jubilant|radiant|blissful|optimistic|"
    r"hopeful|inspired|motivated|energetic|enthusiastic|overjoyed|pleasant|"
    # Hinglish
    r"khush|mast|zabardast|badhiya|shukriya|theek hai|accha|accha hai|"
    r"maja aa raha|acha lag raha|sab theek|kal|ekdum mast|maja|"
    r"i am happy|i feel good|feeling good|feeling great|i am okay|"
    r"life is good|so happy|very happy|really happy|super happy|"
    r"love it|loving it|enjoying|having fun|good mood|great mood)\b",
    re.I,
)

# ── Sad / Depressed ─────────────────────────────────────────────────────────
_SAD = re.compile(
    r"\b(sad|sadness|depressed|depression|hopeless|hopelessness|cry|crying|cried|"
    r"tears|tear|heartbroken|broken|shattered|devastated|miserable|unhappy|"
    r"gloomy|melancholy|grief|grieving|mourning|sorrow|sorrowful|despairing|"
    r"despondent|dejected|downcast|down|low|numb|empty|hollow|lost|"
    r"lonely|alone|isolated|abandoned|rejected|unwanted|unloved|"
    r"miss|missing|miss you|missed|i miss|cannot smile|can't smile|"
    r"no point|pointless|worthless|useless|failure|loser|"
    r"tired of|exhausted|drained|dead inside|feel nothing|"
    # Hinglish
    r"udaas|dukhi|rona|rona aa raha|dil toot gaya|dard|takleef|"
    r"akela|tanha|bemani|kuch nahi chahiye|thak gaya|thak gayi|"
    r"i feel sad|feeling sad|feeling down|feeling low|very sad|so sad|"
    r"really sad|deeply sad|can't stop crying|keep crying|been crying)\b",
    re.I,
)

# ── Angry ────────────────────────────────────────────────────────────────────
_ANG = re.compile(
    r"\b(angry|anger|furious|rage|mad|hate|annoyed|frustrated|frustration|"
    r"pissed|irritated|livid|outraged|enraged|infuriated|fuming|seething|"
    r"disgusted|disgusting|sick of|fed up|fed up with|had enough|"
    r"cant stand|can't stand|cannot stand|i hate|why can't|why won't|"
    r"unfair|not fair|ridiculous|absurd|stupid|idiot|pathetic|"
    r"pisses me off|makes me angry|drives me crazy|lost my temper|"
    # Hinglish
    r"gussa|naraaz|chidchida|ghusa|jalaahat|bakwaas|bekar|ganda|"
    r"i am angry|i am furious|so angry|very angry|really angry|"
    r"how dare|this is wrong|unacceptable|not acceptable)\b",
    re.I,
)

# ── Anxious / Worried ────────────────────────────────────────────────────────
_ANX = re.compile(
    r"\b(anxious|anxiety|worried|worry|worrying|nervous|panic|panicking|"
    r"stress|stressed|stressing|scared|afraid|fear|fearful|frightened|"
    r"overwhelmed|overwhelming|dread|dreading|apprehensive|uneasy|"
    r"on edge|tense|tension|cant breathe|can't breathe|chest tight|"
    r"heart racing|racing heart|shaking|trembling|overthinking|"
    r"what if|so worried|very worried|really worried|been anxious|"
    r"keep thinking|can't stop thinking|racing thoughts|spiraling|"
    r"cant cope|can't cope|cannot cope|losing it|breaking down|"
    # Hinglish
    r"dar|darr|ghabrana|ghabra raha|ghabra gayi|fikar|chinta|"
    r"tension hai|bahut tension|pareshan|tension ho rahi|"
    r"i am scared|i am nervous|i am anxious|feeling anxious|"
    r"feeling nervous|feeling scared|feel anxious|feel worried)\b",
    re.I,
)

# ── Distress / Crisis ─────────────────────────────────────────────────────────
_DISTRESS = re.compile(
    r"\b(help me|please help|desperate|crisis|emergency|breaking point|"
    r"can't take it|cannot take it|can't do this|cannot do this|"
    r"give up|giving up|want to give up|want to die|don't want to live|"
    r"no reason to live|no hope|no way out|end it all|"
    r"self harm|hurt myself|hurt myself|suicidal|not okay|"
    r"losing my mind|going crazy|i'm done|im done|i give up|"
    # Hinglish
    r"bas kar do|bahut hogaya|nahi rehna|khatam karna|"
    r"please help|someone help|help me please)\b",
    re.I,
)

# ── Low mood / Fatigue ────────────────────────────────────────────────────────
_LOW_MOOD = re.compile(
    r"\b(tired|exhausted|drained|worn out|burnt out|burned out|"
    r"no energy|zero energy|no motivation|unmotivated|"
    r"can't get up|can't get out of bed|don't want to do anything|"
    r"nothing matters|don't care anymore|lost interest|"
    r"going through a lot|rough time|rough patch|hard time|"
    r"not myself|not feeling like myself|off|bit off|"
    r"insomnia|can't sleep|cannot sleep|not sleeping|no sleep|"
    r"sleep deprived|haven't slept|bad day|terrible day|worst day|"
    # Hinglish
    r"bahut thaka|thak gaya hoon|neend nahi|sota nahi|soti nahi|"
    r"kuch karne ka mann nahi|kuch achha nahi lag raha)\b",
    re.I,
)

# ── Emotional masking / Suppression ──────────────────────────────────────────
_MASK = re.compile(
    r"\b(i'?m fine|i am fine|i'?m okay|i am okay|i'?m alright|i am alright|"
    r"everything is fine|everything'?s fine|all good|all is good|nothing'?s wrong|"
    r"nothing is wrong|just tired|just a bit tired|don'?t worry|don'?t worry about me|"
    r"not a big deal|just stressed|i'?ll be fine|it'?s nothing|"
    r"just overthinking|maybe i'?m overthinking|i'?ll get through it|"
    r"i'?m used to it|nothing new|same old|just life|"
    # Hinglish
    r"sab theek hai|main theek hoon|koi baat nahi|chhod do|"
    r"tension mat lo|main sambhal lunga|main sambhal lungi)\b",
    re.I,
)

# ── Negation detection ────────────────────────────────────────────────────────
_NEG_PREFIX = re.compile(r"\b(not|never|no|none|don'?t|didn'?t|can'?t|won'?t|"
                          r"isn'?t|wasn'?t|aren'?t|haven'?t|hadn'?t)\s+", re.I)


def _count_negated_pos(text: str) -> int:
    """Count positive words that are negated (e.g. 'not happy', 'not good')."""
    count = 0
    for m in _NEG_PREFIX.finditer(text):
        following = text[m.end():m.end() + 40]
        if _POS.match(following.split()[0] if following.split() else ""):
            count += 1
    return count


def detect(text: str) -> dict:
    t = (text or "").strip()
    if not t:
        return {"label": "neutral", "confidence": 0.0, "scores": {}}

    pos = len(_POS.findall(t))
    sad = len(_SAD.findall(t))
    ang = len(_ANG.findall(t))
    anx = len(_ANX.findall(t))
    dist = len(_DISTRESS.findall(t))
    low = len(_LOW_MOOD.findall(t))
    mask = len(_MASK.findall(t))
    neg_pos = _count_negated_pos(t)

    # Subtract negated positives from pos count
    pos = max(0, pos - neg_pos)
    # Negated positives add to sadness signal
    sad += neg_pos

    # ── Build raw scores ──────────────────────────────────────────────────────
    scores = {
        "distressed": min(1.0, dist * 0.5 + ang * 0.05 + anx * 0.05),
        "angry": min(1.0, ang * 0.35 + (anx * 0.05 if ang > anx else 0)),
        "anxious": min(1.0, anx * 0.32 + low * 0.08),
        "sad": min(1.0, sad * 0.32 + low * 0.06 + neg_pos * 0.15),
        "low_mood": min(1.0, low * 0.30 + sad * 0.08),
        "happy": min(1.0, pos * 0.30),
        "neutral": 0.0,
    }

    # ── Determine initial label ───────────────────────────────────────────────
    if dist >= 1:
        label = "distressed"
    elif ang >= 1 and ang > anx:
        label = "angry"
    elif anx >= 1 and anx >= ang:
        label = "anxious"
    elif sad > pos and sad >= 1:
        label = "sad"
    elif low >= 2 and low > pos:
        label = "low_mood"
    elif pos > (sad + ang + anx + low):
        label = "happy"
    else:
        label = "neutral"

    # Confidence based on total evidence
    total_evidence = pos + sad + ang + anx + dist + low
    confidence = min(1.0, 0.35 + 0.10 * total_evidence)

    # ── VADER refinement (with much more sensitive thresholds) ─────────────────
    if _vader is not None:
        vs = _vader.polarity_scores(t)
        compound = float(vs["compound"])
        neg_score = float(vs["neg"])
        pos_score = float(vs["pos"])

        # Override to sad (was -0.45, now -0.20)
        if label in ("neutral", "low_mood") and compound <= -0.20:
            label = "sad"
            confidence = max(confidence, 0.52)
        # Low mood detection (was -0.20, now -0.08)
        elif label == "neutral" and compound <= -0.08:
            label = "low_mood"
            confidence = max(confidence, 0.45)
        # Override to happy (was 0.42, now 0.25)
        elif label == "neutral" and compound >= 0.25:
            label = "happy"
            confidence = max(confidence, 0.50)

        # Boost anxiety if negative score is high even with mild compound
        if label == "neutral" and neg_score >= 0.25 and anx >= 1:
            label = "anxious"
            confidence = max(confidence, 0.50)

        # Emotional mask detection: says "fine" but sentiment is negative
        if mask >= 1 and compound < -0.05 and label in ("neutral", "happy", "low_mood"):
            label = "emotional_mismatch"
            confidence = max(confidence, 0.55)

        # False positive happy: says happy words but VADER says negative
        if label == "happy" and compound < -0.15:
            label = "emotional_mismatch"
            confidence = max(confidence, 0.55)

        # Distress detection: very negative language
        if compound <= -0.60 and label not in ("angry", "distressed"):
            label = "distressed"
            confidence = max(confidence, 0.62)

        # Angry boost: use negation score
        if label == "neutral" and neg_score >= 0.35 and ang >= 1:
            label = "angry"
            confidence = max(confidence, 0.52)

    # Normalize scores dict
    scores["neutral"] = 0.15 if label != "neutral" else 0.5

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }
