"""
backend/chatbot/intent_classifier.py
──────────────────────────────────────
NLP Intent Classifier for APEX AI Chatbot (v5 — expanded)

Architecture:
    TF-IDF vectorizer (1-3 ngrams) → Logistic Regression
    525 training examples across 10 intents (50-55 per class)

Intents:
    greeting        → rule_based   (instant response)
    bmi_calc        → rule_based   (deterministic formula)
    calorie_calc    → rule_based   (TDEE formula)
    macro_calc      → rule_based   (macro formula)
    workout_plan    → api          (complex, needs LLM)
    nutrition_plan  → api          (complex, needs LLM)
    supplement      → api          (nuanced, needs LLM)
    injury_advice   → api          (medical, always best engine)
    motivation      → local_dl     (conversational)
    general_fitness → local_dl     (open-ended)
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("apex_ai.intent")

ROOT      = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT / "ai_models" / "ml_models"

# ── Intent → Engine mapping ───────────────────────────────────────────────────

INTENT_TO_ENGINE = {
    "greeting":        "rule_based",
    "bmi_calc":        "rule_based",
    "calorie_calc":    "rule_based",
    "macro_calc":      "rule_based",
    "workout_plan":    "api",
    "nutrition_plan":  "api",
    "supplement":      "api",
    "injury_advice":   "api",
    "motivation":      "local_dl",
    "general_fitness": "local_dl",
}

# ── Training Data (525 examples — 50-55 per intent) ───────────────────────────

BUILTIN_TRAINING_DATA = [

    # ── GREETING (50) ─────────────────────────────────────────────────────────
    ("hello", "greeting"), ("hi there", "greeting"), ("hey", "greeting"),
    ("good morning", "greeting"), ("good evening", "greeting"),
    ("good afternoon", "greeting"), ("salam", "greeting"),
    ("marhaba", "greeting"), ("what's up", "greeting"), ("howdy", "greeting"),
    ("hi coach", "greeting"), ("hey coach", "greeting"),
    ("hello apex", "greeting"), ("good night", "greeting"),
    ("morning", "greeting"), ("evening", "greeting"), ("sup", "greeting"),
    ("yo", "greeting"), ("hiya", "greeting"), ("greetings", "greeting"),
    ("hey there", "greeting"), ("hi again", "greeting"),
    ("hello hello", "greeting"), ("good day", "greeting"),
    ("assalam alaikum", "greeting"), ("hey how are you", "greeting"),
    ("hi how are you doing", "greeting"), ("hello i need help", "greeting"),
    ("hi i'm new here", "greeting"), ("hey i just signed up", "greeting"),
    ("good morning coach", "greeting"), ("hi can you help me", "greeting"),
    ("hello i want to start", "greeting"), ("hey i need advice", "greeting"),
    ("hi what can you do", "greeting"), ("hello are you there", "greeting"),
    ("hey wake up", "greeting"), ("hi i'm back", "greeting"),
    ("good to see you", "greeting"), ("hello fitness coach", "greeting"),
    ("hi trainer", "greeting"), ("hey trainer", "greeting"),
    ("hello bot", "greeting"), ("hi bot", "greeting"),
    ("hey apex ai", "greeting"), ("salut", "greeting"),
    ("bonjour", "greeting"), ("ahlan", "greeting"),
    ("sabah el kheir", "greeting"), ("masa el kheir", "greeting"),

    # ── BMI_CALC (50) ─────────────────────────────────────────────────────────
    ("what is my bmi", "bmi_calc"), ("calculate bmi", "bmi_calc"),
    ("how do I calculate body mass index", "bmi_calc"),
    ("am I overweight", "bmi_calc"), ("what is a healthy bmi", "bmi_calc"),
    ("bmi calculator", "bmi_calc"), ("check my bmi", "bmi_calc"),
    ("my bmi please", "bmi_calc"), ("is my weight healthy", "bmi_calc"),
    ("am I obese", "bmi_calc"), ("what does my bmi mean", "bmi_calc"),
    ("bmi chart", "bmi_calc"), ("normal bmi range", "bmi_calc"),
    ("what is bmi", "bmi_calc"), ("tell me my bmi", "bmi_calc"),
    ("how to calculate bmi", "bmi_calc"),
    ("what bmi should I have", "bmi_calc"),
    ("ideal weight for my height", "bmi_calc"),
    ("am I at a healthy weight", "bmi_calc"),
    ("how do I know if I'm overweight", "bmi_calc"),
    ("bmi for 80kg", "bmi_calc"), ("bmi for 75kg 175cm", "bmi_calc"),
    ("what is my ideal weight", "bmi_calc"),
    ("healthy weight range for me", "bmi_calc"),
    ("underweight or overweight", "bmi_calc"),
    ("body mass index calculator", "bmi_calc"),
    ("is 25 bmi bad", "bmi_calc"), ("bmi 30 what does it mean", "bmi_calc"),
    ("what weight should I be for my height", "bmi_calc"),
    ("how much should I weigh", "bmi_calc"),
    ("weight to height ratio", "bmi_calc"),
    ("healthy weight for 175cm", "bmi_calc"), ("am I fat", "bmi_calc"),
    ("how overweight am I", "bmi_calc"),
    ("calculate my body mass", "bmi_calc"), ("bmi formula", "bmi_calc"),
    ("my height is 170 weight is 70 what is bmi", "bmi_calc"),
    ("overweight threshold", "bmi_calc"), ("obesity bmi range", "bmi_calc"),
    ("what is considered overweight", "bmi_calc"),
    ("normal weight range", "bmi_calc"),
    ("bmi categories explained", "bmi_calc"),
    ("how do doctors calculate bmi", "bmi_calc"),
    ("check if I am at healthy weight", "bmi_calc"),
    ("what does bmi of 27 mean", "bmi_calc"),
    ("bmi 22 is that good", "bmi_calc"),
    ("my bmi is 28 is that bad", "bmi_calc"),
    ("bmi scale for adults", "bmi_calc"), ("ideal bmi for men", "bmi_calc"),
    ("ideal bmi for women", "bmi_calc"),

    # ── CALORIE_CALC (55) ─────────────────────────────────────────────────────
    ("how many calories should I eat", "calorie_calc"),
    ("what is my tdee", "calorie_calc"),
    ("daily calorie needs", "calorie_calc"),
    ("caloric deficit for weight loss", "calorie_calc"),
    ("how many calories to lose weight", "calorie_calc"),
    ("maintenance calories", "calorie_calc"),
    ("calculate my calories", "calorie_calc"),
    ("how many calories do I need", "calorie_calc"),
    ("what is my daily calorie intake", "calorie_calc"),
    ("calorie requirement for me", "calorie_calc"),
    ("how much should I eat to lose weight", "calorie_calc"),
    ("calories to gain muscle", "calorie_calc"),
    ("bulking calories", "calorie_calc"), ("cutting calories", "calorie_calc"),
    ("calorie surplus for bulking", "calorie_calc"),
    ("calorie deficit how much", "calorie_calc"),
    ("500 calorie deficit", "calorie_calc"),
    ("how many calories to maintain weight", "calorie_calc"),
    ("total daily energy expenditure", "calorie_calc"),
    ("bmr calculation", "calorie_calc"),
    ("basal metabolic rate", "calorie_calc"),
    ("how many calories does my body burn", "calorie_calc"),
    ("what is my bmr", "calorie_calc"),
    ("resting metabolic rate", "calorie_calc"),
    ("calories needed per day", "calorie_calc"),
    ("daily energy needs", "calorie_calc"),
    ("how many calories for fat loss", "calorie_calc"),
    ("calorie goal for weight loss", "calorie_calc"),
    ("how many kcal per day", "calorie_calc"),
    ("how much food should I eat", "calorie_calc"),
    ("calories to eat to lose 1kg per week", "calorie_calc"),
    ("calories for muscle building", "calorie_calc"),
    ("what calorie deficit should I have", "calorie_calc"),
    ("1200 calories is that enough", "calorie_calc"),
    ("1500 calories diet", "calorie_calc"),
    ("2000 calories a day", "calorie_calc"),
    ("should I eat 1800 calories", "calorie_calc"),
    ("calorie counting help", "calorie_calc"),
    ("how to calculate my calorie needs", "calorie_calc"),
    ("am I eating enough calories", "calorie_calc"),
    ("am I eating too many calories", "calorie_calc"),
    ("calorie intake for sedentary person", "calorie_calc"),
    ("calorie intake for active person", "calorie_calc"),
    ("how many calories for 80kg male", "calorie_calc"),
    ("calorie needs based on activity level", "calorie_calc"),
    ("how many calories does exercise burn", "calorie_calc"),
    ("net calories after exercise", "calorie_calc"),
    ("what is a safe calorie deficit", "calorie_calc"),
    ("minimum calories to eat", "calorie_calc"),
    ("starvation mode calories", "calorie_calc"),
    ("calorie cycling", "calorie_calc"), ("refeed day calories", "calorie_calc"),
    ("how many calories on rest day", "calorie_calc"),
    ("training day vs rest day calories", "calorie_calc"),
    ("calorie targets for my goals", "calorie_calc"),

    # ── MACRO_CALC (52) ───────────────────────────────────────────────────────
    ("how much protein should I eat", "macro_calc"),
    ("what are my macros", "macro_calc"),
    ("protein carbs fat ratio", "macro_calc"),
    ("macro breakdown", "macro_calc"),
    ("how much protein per day", "macro_calc"), ("keto macros", "macro_calc"),
    ("iifym", "macro_calc"), ("flexible dieting macros", "macro_calc"),
    ("macro targets for me", "macro_calc"),
    ("calculate my macros", "macro_calc"),
    ("how many grams of protein do I need", "macro_calc"),
    ("protein per kg of body weight", "macro_calc"),
    ("1g protein per pound", "macro_calc"),
    ("2g protein per kg", "macro_calc"),
    ("protein intake for muscle gain", "macro_calc"),
    ("carbohydrate intake", "macro_calc"),
    ("how many carbs should I eat", "macro_calc"),
    ("low carb diet macros", "macro_calc"),
    ("high protein diet plan", "macro_calc"),
    ("fat intake per day", "macro_calc"),
    ("how much fat should I eat", "macro_calc"),
    ("macronutrient ratio for fat loss", "macro_calc"),
    ("macros for bulking", "macro_calc"), ("macros for cutting", "macro_calc"),
    ("macros for maintenance", "macro_calc"),
    ("protein fat carb split", "macro_calc"),
    ("40 30 30 macro split", "macro_calc"),
    ("how to calculate my protein needs", "macro_calc"),
    ("lean bulk macros", "macro_calc"),
    ("macro split for body recomposition", "macro_calc"),
    ("how much protein to build muscle", "macro_calc"),
    ("minimum protein intake", "macro_calc"),
    ("optimal protein intake", "macro_calc"),
    ("protein recommendation for athlete", "macro_calc"),
    ("daily protein goal", "macro_calc"),
    ("how much protein to preserve muscle", "macro_calc"),
    ("protein for weight loss", "macro_calc"),
    ("what is a macro", "macro_calc"),
    ("explain macronutrients", "macro_calc"),
    ("carb cycling macros", "macro_calc"),
    ("low carb high protein macros", "macro_calc"),
    ("how many grams of carbs", "macro_calc"),
    ("good fat sources", "macro_calc"), ("healthy fats intake", "macro_calc"),
    ("saturated fat limit", "macro_calc"),
    ("omega 3 intake recommendation", "macro_calc"),
    ("fiber intake per day", "macro_calc"),
    ("how much fiber should I eat", "macro_calc"),
    ("sugar intake limit", "macro_calc"),
    ("protein shake count towards macros", "macro_calc"),
    ("track macros how to", "macro_calc"), ("macro calculator", "macro_calc"),

    # ── WORKOUT_PLAN (55) ─────────────────────────────────────────────────────
    ("give me a workout plan", "workout_plan"),
    ("what exercises should I do", "workout_plan"),
    ("build muscle workout", "workout_plan"),
    ("weekly training program", "workout_plan"),
    ("best exercises for beginners", "workout_plan"),
    ("chest workout routine", "workout_plan"),
    ("leg day exercises", "workout_plan"),
    ("full body workout plan", "workout_plan"),
    ("create a workout schedule for me", "workout_plan"),
    ("gym routine for fat loss", "workout_plan"),
    ("push pull legs routine", "workout_plan"),
    ("upper lower split", "workout_plan"),
    ("5 day workout split", "workout_plan"),
    ("3 day workout plan", "workout_plan"),
    ("4 day workout plan", "workout_plan"),
    ("beginner gym program", "workout_plan"),
    ("intermediate workout program", "workout_plan"),
    ("advanced training program", "workout_plan"),
    ("hypertrophy program", "workout_plan"),
    ("strength training program", "workout_plan"),
    ("powerlifting program", "workout_plan"),
    ("home workout plan no equipment", "workout_plan"),
    ("bodyweight workout routine", "workout_plan"),
    ("dumbbell only workout plan", "workout_plan"),
    ("back workout exercises", "workout_plan"),
    ("shoulder workout routine", "workout_plan"),
    ("arm workout biceps triceps", "workout_plan"),
    ("glute workout program", "workout_plan"),
    ("ab workout routine", "workout_plan"),
    ("cardio and weights program", "workout_plan"),
    ("12 week transformation program", "workout_plan"),
    ("6 day ppl program", "workout_plan"),
    ("how many days should I train", "workout_plan"),
    ("what muscles to train together", "workout_plan"),
    ("muscle group split", "workout_plan"),
    ("exercise selection for beginners", "workout_plan"),
    ("compound exercises list", "workout_plan"),
    ("best exercises for fat loss", "workout_plan"),
    ("best exercises for muscle gain", "workout_plan"),
    ("workout plan for women", "workout_plan"),
    ("workout plan for men", "workout_plan"),
    ("plan my workouts for the week", "workout_plan"),
    ("training schedule help", "workout_plan"),
    ("how to structure my workouts", "workout_plan"),
    ("sets and reps for muscle gain", "workout_plan"),
    ("how many sets per muscle group", "workout_plan"),
    ("how many reps for strength", "workout_plan"),
    ("progressive overload program", "workout_plan"),
    ("olympic lifting program", "workout_plan"),
    ("crossfit style workout", "workout_plan"),
    ("circuit training program", "workout_plan"),
    ("HIIT workout plan", "workout_plan"),
    ("sprint training program", "workout_plan"),
    ("calisthenics program", "workout_plan"),
    ("gym program for absolute beginner", "workout_plan"),

    # ── NUTRITION_PLAN (52) ───────────────────────────────────────────────────
    ("what should I eat to lose weight", "nutrition_plan"),
    ("meal plan for muscle gain", "nutrition_plan"),
    ("healthy diet plan", "nutrition_plan"),
    ("what foods to avoid", "nutrition_plan"),
    ("intermittent fasting schedule", "nutrition_plan"),
    ("best diet for fat loss", "nutrition_plan"),
    ("give me a meal plan", "nutrition_plan"),
    ("weekly meal plan", "nutrition_plan"),
    ("what to eat for breakfast", "nutrition_plan"),
    ("what to eat before workout", "nutrition_plan"),
    ("what to eat after workout", "nutrition_plan"),
    ("post workout meal ideas", "nutrition_plan"),
    ("pre workout food", "nutrition_plan"),
    ("healthy meal ideas", "nutrition_plan"),
    ("high protein meals", "nutrition_plan"),
    ("meal prep ideas", "nutrition_plan"),
    ("diet plan for me", "nutrition_plan"),
    ("bulking meal plan", "nutrition_plan"),
    ("cutting diet plan", "nutrition_plan"),
    ("vegan muscle building diet", "nutrition_plan"),
    ("vegetarian diet for fitness", "nutrition_plan"),
    ("keto diet plan", "nutrition_plan"),
    ("low carb meal plan", "nutrition_plan"),
    ("mediterranean diet plan", "nutrition_plan"),
    ("clean eating meal plan", "nutrition_plan"),
    ("foods that build muscle", "nutrition_plan"),
    ("foods that burn fat", "nutrition_plan"),
    ("best foods for athletes", "nutrition_plan"),
    ("nutrition plan for weight loss", "nutrition_plan"),
    ("healthy eating plan", "nutrition_plan"),
    ("diet advice for gym", "nutrition_plan"),
    ("what should I eat today", "nutrition_plan"),
    ("design my diet", "nutrition_plan"),
    ("foods high in protein list", "nutrition_plan"),
    ("low calorie high protein foods", "nutrition_plan"),
    ("best carb sources for athletes", "nutrition_plan"),
    ("healthy fat sources for diet", "nutrition_plan"),
    ("meal timing for muscle gain", "nutrition_plan"),
    ("how many meals per day", "nutrition_plan"),
    ("3 meals vs 6 meals", "nutrition_plan"),
    ("should I eat breakfast", "nutrition_plan"),
    ("diet for lean bulking", "nutrition_plan"),
    ("how to eat clean", "nutrition_plan"),
    ("nutrition for beginners", "nutrition_plan"),
    ("eating plan for gym", "nutrition_plan"),
    ("food plan for weight loss", "nutrition_plan"),
    ("meal schedule for fat loss", "nutrition_plan"),
    ("meal ideas for muscle building", "nutrition_plan"),
    ("what foods should I eat daily", "nutrition_plan"),
    ("high protein low calorie diet", "nutrition_plan"),
    ("diet tips for gym beginners", "nutrition_plan"),
    ("foods to eat to lose belly fat", "nutrition_plan"),

    # ── SUPPLEMENT (52) ───────────────────────────────────────────────────────
    ("should I take creatine", "supplement"),
    ("best protein powder", "supplement"),
    ("pre workout supplement", "supplement"),
    ("is creatine safe", "supplement"),
    ("whey vs casein protein", "supplement"),
    ("omega 3 benefits", "supplement"),
    ("what supplements should I take", "supplement"),
    ("best supplements for beginners", "supplement"),
    ("do I need supplements", "supplement"),
    ("creatine monohydrate dosage", "supplement"),
    ("how to take creatine", "supplement"),
    ("creatine loading phase", "supplement"),
    ("bcaa supplement", "supplement"), ("are bcaas worth it", "supplement"),
    ("beta alanine supplement", "supplement"),
    ("caffeine pre workout", "supplement"),
    ("vitamin d supplement", "supplement"),
    ("magnesium supplement", "supplement"),
    ("zinc supplement for testosterone", "supplement"),
    ("fish oil supplement", "supplement"),
    ("glutamine supplement", "supplement"),
    ("protein powder recommendation", "supplement"),
    ("whey protein isolate vs concentrate", "supplement"),
    ("plant based protein powder", "supplement"),
    ("vegan protein supplement", "supplement"),
    ("best post workout supplement", "supplement"),
    ("supplement stack for muscle gain", "supplement"),
    ("fat burner supplement", "supplement"),
    ("are fat burners effective", "supplement"),
    ("testosterone booster supplement", "supplement"),
    ("natural testosterone supplements", "supplement"),
    ("collagen supplement for joints", "supplement"),
    ("glucosamine for knees", "supplement"),
    ("ashwagandha benefits", "supplement"),
    ("melatonin for sleep recovery", "supplement"),
    ("electrolyte supplement", "supplement"),
    ("when to take protein shake", "supplement"),
    ("protein shake before or after workout", "supplement"),
    ("how much creatine per day", "supplement"),
    ("creatine side effects", "supplement"),
    ("does creatine cause bloating", "supplement"),
    ("best brand of creatine", "supplement"),
    ("how long to take creatine", "supplement"),
    ("cycle creatine or take daily", "supplement"),
    ("pre workout side effects", "supplement"),
    ("caffeine tolerance supplement", "supplement"),
    ("best time to take vitamin d", "supplement"),
    ("iron supplement for athletes", "supplement"),
    ("b12 supplement for vegans", "supplement"),
    ("protein powder for weight loss", "supplement"),
    ("protein powder for muscle gain", "supplement"),
    ("supplement timing guide", "supplement"),

    # ── INJURY_ADVICE (52) ────────────────────────────────────────────────────
    ("my knee hurts when squatting", "injury_advice"),
    ("lower back pain from deadlifts", "injury_advice"),
    ("shoulder injury from bench press", "injury_advice"),
    ("how to avoid injury", "injury_advice"),
    ("muscle strain recovery", "injury_advice"),
    ("rotator cuff pain", "injury_advice"),
    ("wrist pain lifting", "injury_advice"),
    ("elbow pain curls", "injury_advice"),
    ("hip flexor pain", "injury_advice"),
    ("shin splints from running", "injury_advice"),
    ("IT band syndrome", "injury_advice"),
    ("tennis elbow gym", "injury_advice"),
    ("golfer's elbow", "injury_advice"),
    ("neck pain from overhead press", "injury_advice"),
    ("knee clicking when squatting", "injury_advice"),
    ("shoulder clicking when pressing", "injury_advice"),
    ("muscle pull recovery time", "injury_advice"),
    ("how long does muscle strain take to heal", "injury_advice"),
    ("can I train with sore muscles", "injury_advice"),
    ("doms vs injury", "injury_advice"),
    ("how to train around injury", "injury_advice"),
    ("exercise with bad knees", "injury_advice"),
    ("workout with lower back pain", "injury_advice"),
    ("bad shoulder exercises to avoid", "injury_advice"),
    ("knee injury gym alternatives", "injury_advice"),
    ("how to warm up to prevent injury", "injury_advice"),
    ("injury prevention tips", "injury_advice"),
    ("pain after deadlift", "injury_advice"),
    ("sore lower back after squats", "injury_advice"),
    ("impingement syndrome shoulder", "injury_advice"),
    ("torn muscle symptoms", "injury_advice"),
    ("strain vs sprain difference", "injury_advice"),
    ("should I train if I have pain", "injury_advice"),
    ("rest or train through pain", "injury_advice"),
    ("ice or heat for muscle injury", "injury_advice"),
    ("how to reduce muscle soreness", "injury_advice"),
    ("foam rolling for recovery", "injury_advice"),
    ("stretching for injury prevention", "injury_advice"),
    ("tight hip flexors fix", "injury_advice"),
    ("patellar tendonitis gym", "injury_advice"),
    ("achilles tendon pain lifting", "injury_advice"),
    ("calf strain recovery", "injury_advice"),
    ("groin strain exercise", "injury_advice"),
    ("bicep tendon pain", "injury_advice"),
    ("tricep pain from dips", "injury_advice"),
    ("plantar fasciitis and exercise", "injury_advice"),
    ("back injury prevention lifting", "injury_advice"),
    ("safe exercises with herniated disc", "injury_advice"),
    ("neck strain from gym", "injury_advice"),
    ("overuse injury symptoms", "injury_advice"),
    ("how to deload after injury", "injury_advice"),
    ("returning to gym after injury", "injury_advice"),

    # ── MOTIVATION (52) ───────────────────────────────────────────────────────
    ("I don't feel like working out", "motivation"),
    ("how to stay motivated", "motivation"),
    ("I want to give up", "motivation"),
    ("not seeing results", "motivation"),
    ("tips to stay consistent", "motivation"),
    ("how to build gym habit", "motivation"),
    ("I keep skipping gym", "motivation"),
    ("lost my motivation", "motivation"),
    ("can't be bothered to workout", "motivation"),
    ("no energy to train", "motivation"),
    ("I hate going to the gym", "motivation"),
    ("how to enjoy working out", "motivation"),
    ("fitness motivation tips", "motivation"),
    ("how to be consistent with diet", "motivation"),
    ("I always quit after 2 weeks", "motivation"),
    ("how to not give up", "motivation"),
    ("plateau and feeling stuck", "motivation"),
    ("no progress in months", "motivation"),
    ("feeling discouraged about fitness", "motivation"),
    ("is it worth it to work out", "motivation"),
    ("gym anxiety tips", "motivation"),
    ("scared to go to gym", "motivation"),
    ("embarrassed at gym", "motivation"),
    ("how to get into fitness routine", "motivation"),
    ("build discipline for gym", "motivation"),
    ("gym accountability", "motivation"),
    ("how to track progress for motivation", "motivation"),
    ("before and after photos motivation", "motivation"),
    ("working out alone vs with partner", "motivation"),
    ("gym partner motivation", "motivation"),
    ("morning workout motivation", "motivation"),
    ("evening workout motivation", "motivation"),
    ("I'm too tired to workout", "motivation"),
    ("too busy to workout motivation", "motivation"),
    ("work life balance fitness", "motivation"),
    ("stressed and can't workout", "motivation"),
    ("workout when depressed", "motivation"),
    ("mental health and exercise", "motivation"),
    ("exercise helps anxiety", "motivation"),
    ("benefits of working out mentally", "motivation"),
    ("how long until I see results", "motivation"),
    ("results taking too long", "motivation"),
    ("slow progress is discouraging", "motivation"),
    ("I feel like I'm not improving", "motivation"),
    ("comparing myself to others at gym", "motivation"),
    ("stop comparing fitness progress", "motivation"),
    ("how to celebrate small wins", "motivation"),
    ("fitness journey mindset", "motivation"),
    ("why am I not motivated anymore", "motivation"),
    ("getting back on track fitness", "motivation"),
    ("restart fitness journey", "motivation"),
    ("consistency over perfection fitness", "motivation"),

    # ── GENERAL_FITNESS (55) ─────────────────────────────────────────────────
    ("how do I lose belly fat", "general_fitness"),
    ("best cardio for fat loss", "general_fitness"),
    ("how long to see results", "general_fitness"),
    ("should I do cardio or weights", "general_fitness"),
    ("how to track progress", "general_fitness"),
    ("what is progressive overload", "general_fitness"),
    ("how much sleep do I need", "general_fitness"),
    ("recovery tips after workout", "general_fitness"),
    ("how to lose weight fast", "general_fitness"),
    ("how to gain weight", "general_fitness"),
    ("body recomposition tips", "general_fitness"),
    ("how to get abs", "general_fitness"),
    ("how to get a six pack", "general_fitness"),
    ("how to build broad shoulders", "general_fitness"),
    ("how to get bigger arms", "general_fitness"),
    ("how to build a bigger chest", "general_fitness"),
    ("how to grow glutes", "general_fitness"),
    ("how to lose thigh fat", "general_fitness"),
    ("best exercise for fat burning", "general_fitness"),
    ("how to speed up metabolism", "general_fitness"),
    ("does muscle burn more calories", "general_fitness"),
    ("cardio vs weights for fat loss", "general_fitness"),
    ("how many steps per day", "general_fitness"),
    ("walking for weight loss", "general_fitness"),
    ("running vs cycling for fitness", "general_fitness"),
    ("how to improve endurance", "general_fitness"),
    ("how to increase stamina", "general_fitness"),
    ("fitness for beginners tips", "general_fitness"),
    ("gym tips for beginners", "general_fitness"),
    ("how to use gym equipment", "general_fitness"),
    ("what to do first day at gym", "general_fitness"),
    ("gym etiquette", "general_fitness"),
    ("how often should I change workout", "general_fitness"),
    ("muscle confusion myth", "general_fitness"),
    ("how to measure body fat", "general_fitness"),
    ("body fat percentage healthy range", "general_fitness"),
    ("how to lose body fat percentage", "general_fitness"),
    ("what is a deload week", "general_fitness"),
    ("when to take rest day", "general_fitness"),
    ("active recovery ideas", "general_fitness"),
    ("stretching routine", "general_fitness"),
    ("flexibility training tips", "general_fitness"),
    ("yoga for muscle recovery", "general_fitness"),
    ("how to warm up properly", "general_fitness"),
    ("cool down after workout", "general_fitness"),
    ("how to improve posture", "general_fitness"),
    ("posture exercises", "general_fitness"),
    ("beginner fitness goals", "general_fitness"),
    ("realistic fitness goals", "general_fitness"),
    ("how to set fitness goals", "general_fitness"),
    ("smart fitness goals", "general_fitness"),
    ("how to measure fitness progress", "general_fitness"),
    ("fitness tracker benefits", "general_fitness"),
    ("heart rate training zones", "general_fitness"),
    ("zone 2 cardio benefits", "general_fitness"),
]


class IntentClassifier:
    """
    TF-IDF (1-3 ngrams) + Logistic Regression intent classifier.
    525 training examples, 50-55 per class.
    Singleton — trains once on first use.
    """

    _instance   = None
    _model      = None
    _vectorizer = None
    _trained    = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not IntentClassifier._trained:
            self._load_or_train()

    def _load_or_train(self):
        clf_path = MODEL_DIR / "intent_classifier.pkl"
        vec_path = MODEL_DIR / "intent_vectorizer.pkl"
        try:
            import joblib
            if clf_path.exists() and vec_path.exists():
                IntentClassifier._model      = joblib.load(clf_path)
                IntentClassifier._vectorizer = joblib.load(vec_path)
                IntentClassifier._trained    = True
                logger.info("Intent classifier loaded from disk")
                return
        except Exception as e:
            logger.warning(f"Could not load saved model: {e}")
        self._train()

    def _train(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            texts  = [t for t, _ in BUILTIN_TRAINING_DATA]
            labels = [l for _, l in BUILTIN_TRAINING_DATA]

            vec = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=8000,
                sublinear_tf=True,
                analyzer="word",
                min_df=1,
            )
            X = vec.fit_transform(texts)

            clf = LogisticRegression(
                max_iter=1000, C=2.0,
                random_state=42, solver="lbfgs",
                
            )
            clf.fit(X, labels)

            IntentClassifier._vectorizer = vec
            IntentClassifier._model      = clf
            IntentClassifier._trained    = True
            logger.info(f"Intent classifier trained on {len(texts)} examples")
        except ImportError:
            logger.warning("scikit-learn not available — intent classifier disabled")
        except Exception as e:
            logger.warning(f"Training failed: {e}")

    def is_trained(self) -> bool:
        return IntentClassifier._trained

    def classify(self, message: str) -> dict:
        if not IntentClassifier._trained or IntentClassifier._model is None:
            return {"intent": "general_fitness", "confidence": 0.5, "engine": "local_dl"}
        try:
            import numpy as np
            X      = IntentClassifier._vectorizer.transform([message.lower()])
            proba  = IntentClassifier._model.predict_proba(X)[0]
            idx    = int(np.argmax(proba))
            intent = IntentClassifier._model.classes_[idx]
            conf   = float(proba[idx])
            engine = INTENT_TO_ENGINE.get(intent, "local_dl")
            return {"intent": intent, "confidence": round(conf, 3), "engine": engine}
        except Exception as e:
            logger.warning(f"Classification error: {e}")
            return {"intent": "general_fitness", "confidence": 0.5, "engine": "local_dl"}

    def classify_batch(self, messages: list) -> list:
        return [self.classify(m) for m in messages]

    def save(self, model_dir: Optional[Path] = None):
        if not IntentClassifier._trained:
            return
        try:
            import joblib
            d = model_dir or MODEL_DIR
            d.mkdir(parents=True, exist_ok=True)
            joblib.dump(IntentClassifier._model,      d / "intent_classifier.pkl")
            joblib.dump(IntentClassifier._vectorizer, d / "intent_vectorizer.pkl")
            logger.info(f"Intent classifier saved to {d}")
        except Exception as e:
            logger.error(f"Save failed: {e}")

    def evaluate(self) -> dict:
        """Quick cross-validation score on training data."""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            import numpy as np

            texts  = [t for t, _ in BUILTIN_TRAINING_DATA]
            labels = [l for _, l in BUILTIN_TRAINING_DATA]
            vec    = TfidfVectorizer(ngram_range=(1,3), max_features=8000, sublinear_tf=True)
            X      = vec.fit_transform(texts)
            clf    = LogisticRegression(max_iter=1000, C=2.0, random_state=42)
            scores = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
            return {"cv_accuracy_mean": round(float(scores.mean()), 3),
                    "cv_accuracy_std":  round(float(scores.std()), 3),
                    "n_samples": len(texts)}
        except Exception as e:
            return {"error": str(e)}
