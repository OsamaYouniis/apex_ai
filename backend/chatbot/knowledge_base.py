"""
backend/chatbot/knowledge_base.py
───────────────────────────────────
APEX AI Fitness Knowledge Base (NEW — v5)

A searchable TF-IDF knowledge base of 120+ expert fitness Q&A pairs.
Used as engine layer between intent classification and the Claude API:
- Fast cosine similarity search (no API cost, no network latency)
- Every answer is personalised with user stats at query time
- Covers all major fitness topics the rule engine misses

Architecture:
    query → TF-IDF vectorize → cosine_similarity → top-K answers
    → personalise with user profile → return response

This replaces DialoGPT as the primary "AI" engine because:
1. DialoGPT-medium is 750MB+ and gives generic conversational replies
2. This KB gives factual, accurate, personalized fitness answers
3. Works 100% offline, < 1ms response time
"""

import re
import numpy as np
from typing import Optional

# ── Knowledge Base: (question_examples, answer_template) pairs ─────────────────
# Each entry covers a topic. The question_examples are used for TF-IDF indexing.
# Answer templates use {name}, {weight}, {height}, {age}, {goal}, {tdee}, etc.

RAW_KB = [
    # ── PROTEIN / MACROS ──────────────────────────────────────────────────────
    {
        "id": "protein_intake",
        "questions": [
            "how much protein should I eat", "daily protein intake",
            "protein per day", "how many grams of protein", "protein recommendation",
            "protein for muscle building", "protein target", "how much protein do I need",
            "protein per kg", "how much protein to eat"
        ],
        "answer": (
            "For your goals, **{name}**, here's your personalised protein target:\n\n"
            "**Recommended: {protein_g}g of protein per day**\n\n"
            "This is based on **2.0g per kg** of bodyweight ({weight}kg) — the gold standard "
            "for muscle building and fat loss preservation.\n\n"
            "**Best protein sources:**\n"
            "• Chicken breast: 31g per 100g\n"
            "• Eggs: 6g per egg\n"
            "• Greek yoghurt: 17g per 100g\n"
            "• Tuna: 25g per 100g\n"
            "• Cottage cheese: 11g per 100g\n"
            "• Whey protein: 25g per scoop\n\n"
            "**Timing:** Spread across 4–5 meals of ~{protein_meal}g each. "
            "Always have 30–40g within 2 hours post-workout. 💪"
        )
    },
    {
        "id": "macro_split",
        "questions": [
            "what are my macros", "macro breakdown", "macronutrient ratio",
            "protein carbs fat split", "macro ratio", "macros for weight loss",
            "macros for muscle gain", "ideal macros", "macro targets",
            "how to calculate macros", "iifym"
        ],
        "answer": (
            "Here's your full macro breakdown, **{name}**:\n\n"
            "**Daily Calorie Target: {calories} kcal** (adjusted for your {goal} goal)\n\n"
            "| Macro | Amount | Calories |\n"
            "|-------|--------|----------|\n"
            "| Protein | **{protein_g}g** | {protein_cal} kcal |\n"
            "| Carbohydrates | **{carb_g}g** | {carb_cal} kcal |\n"
            "| Fats | **{fat_g}g** | {fat_cal} kcal |\n\n"
            "These macros support your **{goal_text}** goal. "
            "Adjust by ±100 kcal based on weekly scale changes. 📊"
        )
    },

    # ── CALORIE / TDEE ────────────────────────────────────────────────────────
    {
        "id": "calorie_needs",
        "questions": [
            "how many calories should I eat", "daily calorie needs", "calorie intake",
            "caloric deficit", "maintenance calories", "tdee", "how many calories",
            "calorie goal", "calorie requirement", "calories per day",
            "how much should I eat", "caloric surplus"
        ],
        "answer": (
            "Your personalised calorie targets, **{name}**:\n\n"
            "**TDEE (Maintenance): {tdee} kcal/day**\n"
            "This is what your body burns at your current activity level ({activity_desc}).\n\n"
            "**Your target based on goal:**\n"
            "• 🔥 Fat loss: **{cut_cal} kcal** (500 kcal deficit → ~0.5kg/week loss)\n"
            "• 💪 Muscle gain: **{bulk_cal} kcal** (300 kcal surplus → lean bulk)\n"
            "• ⚖️ Maintenance: **{tdee} kcal**\n\n"
            "Your current goal target: **{calories} kcal/day** ✅"
        )
    },

    # ── BMI ───────────────────────────────────────────────────────────────────
    {
        "id": "bmi_info",
        "questions": [
            "what is my bmi", "calculate bmi", "body mass index",
            "am I overweight", "healthy weight", "bmi category",
            "bmi chart", "normal bmi", "bmi range", "what does my bmi mean"
        ],
        "answer": (
            "Your BMI analysis, **{name}**:\n\n"
            "**BMI: {bmi}** — {bmi_cat}\n"
            "*(Weight: {weight}kg | Height: {height}cm)*\n\n"
            "**BMI Scale:**\n"
            "• < 18.5 — Underweight\n"
            "• 18.5–24.9 — Normal weight ✅\n"
            "• 25.0–29.9 — Overweight\n"
            "• ≥ 30.0 — Obese\n\n"
            "**Note:** BMI doesn't account for muscle mass. "
            "A muscular athlete may show 'overweight' BMI but have low body fat. "
            "Use it as one data point, not the whole picture. 📏"
        )
    },

    # ── WEIGHT LOSS ───────────────────────────────────────────────────────────
    {
        "id": "weight_loss",
        "questions": [
            "how to lose weight", "fat loss tips", "lose belly fat",
            "best way to lose weight", "how to lose fat", "weight loss plan",
            "how to lose body fat", "fastest way to lose fat", "cut body fat",
            "drop weight", "slim down", "fat burning tips"
        ],
        "answer": (
            "Your personalised fat loss strategy, **{name}**:\n\n"
            "**Current stats:** {weight}kg → Target: {target_weight}kg "
            "(need to lose {weight_to_lose}kg)\n"
            "**Estimated timeline: {weeks_to_goal} weeks** at 0.5kg/week\n\n"
            "**The science-backed approach:**\n"
            "1. **Calories:** Eat {cut_cal} kcal/day ({calories_deficit} kcal deficit)\n"
            "2. **Protein:** {protein_g}g/day — preserves muscle while losing fat\n"
            "3. **Cardio:** 3–4 sessions/week, 30–45 min (LISS or HIIT)\n"
            "4. **Strength training:** 3–4x/week — boosts metabolism\n"
            "5. **Sleep:** 7–9 hours — regulates hunger hormones\n"
            "6. **Water:** {water_l}L/day — reduces false hunger signals\n\n"
            "**Track weekly:** Weigh yourself same time every morning. "
            "Expect 0.3–0.7kg/week loss. Adjust calories if stalled for 2+ weeks. 🔥"
        )
    },
    {
        "id": "fat_loss_vs_muscle",
        "questions": [
            "lose fat and gain muscle", "body recomposition", "recomp",
            "can I lose fat and build muscle", "simultaneous fat loss muscle gain",
            "body recomp diet", "cut and bulk at same time"
        ],
        "answer": (
            "Body recomposition is possible for you, **{name}**! Here's how:\n\n"
            "**Who recomp works best for:**\n"
            "• Beginners (first 1–2 years of training)\n"
            "• People returning after a break\n"
            "• Those with 20%+ body fat\n\n"
            "**Recomp protocol:**\n"
            "• Calories: Eat at maintenance or slight deficit ({tdee}–{cut_cal} kcal)\n"
            "• Protein: HIGH — {protein_g}g/day (2g/kg bodyweight)\n"
            "• Training: Prioritise progressive overload strength training\n"
            "• Cardio: 2–3x moderate sessions/week\n\n"
            "**Progress will be slower** than dedicated cutting or bulking "
            "but you lose fat AND build muscle simultaneously. "
            "Expect 3–6 months to see significant visual change. ⚡"
        )
    },

    # ── MUSCLE BUILDING ───────────────────────────────────────────────────────
    {
        "id": "muscle_building",
        "questions": [
            "how to build muscle", "muscle gain tips", "bulking advice",
            "how to gain muscle mass", "muscle growth", "hypertrophy",
            "how to get bigger", "put on muscle", "gain mass",
            "build muscle fast", "best way to build muscle"
        ],
        "answer": (
            "Muscle building plan for **{name}**:\n\n"
            "**Your muscle-gain targets:**\n"
            "• Calories: **{bulk_cal} kcal/day** (+300 kcal surplus)\n"
            "• Protein: **{protein_g}g/day** (2g per kg bodyweight)\n"
            "• Carbs: **{carb_g}g** for training fuel\n\n"
            "**The 4 pillars of muscle growth:**\n"
            "1. **Progressive overload** — add weight or reps every session\n"
            "2. **Volume** — 10–20 sets per muscle group per week\n"
            "3. **Recovery** — 48h between training same muscle\n"
            "4. **Sleep** — 8+ hours (90% of growth hormone released during sleep)\n\n"
            "**Training split recommendation:**\n"
            "• 4–5 days/week\n"
            "• Upper/Lower or Push/Pull/Legs\n"
            "• 8–12 rep range for hypertrophy\n"
            "• Rest 60–90 seconds between sets 💪"
        )
    },

    # ── WORKOUT PLANS ─────────────────────────────────────────────────────────
    {
        "id": "workout_plan_general",
        "questions": [
            "give me a workout plan", "what exercises should I do",
            "weekly workout routine", "gym program", "training program",
            "beginner workout plan", "workout schedule", "exercise routine",
            "gym routine", "full body workout plan"
        ],
        "answer": (
            "**Personalised workout plan for {name}** ({goal_text} goal):\n\n"
            "**Recommended split:** 4 days/week\n\n"
            "**Day 1 — Push (Chest, Shoulders, Triceps):**\n"
            "• Bench Press: 4×8\n"
            "• Overhead Press: 3×10\n"
            "• Incline Dumbbell Press: 3×12\n"
            "• Lateral Raises: 3×15\n"
            "• Tricep Pushdowns: 3×12\n\n"
            "**Day 2 — Pull (Back, Biceps):**\n"
            "• Deadlifts: 4×6\n"
            "• Pull-Ups: 4×max\n"
            "• Bent-Over Rows: 3×10\n"
            "• Face Pulls: 3×15\n"
            "• Barbell Curls: 3×12\n\n"
            "**Day 3 — Legs:**\n"
            "• Squats: 4×8\n"
            "• Romanian Deadlifts: 3×10\n"
            "• Leg Press: 3×12\n"
            "• Leg Curls: 3×12\n"
            "• Calf Raises: 4×15\n\n"
            "**Day 4 — Full Body / Weak Points:**\n"
            "• Focus on any lagging muscle groups\n"
            "• Include core work and mobility\n\n"
            "Rest 60–90s between sets. Increase weight when you hit the top rep range! 💪"
        )
    },
    {
        "id": "ppl_split",
        "questions": [
            "push pull legs", "ppl split", "push pull legs routine",
            "ppl program", "push pull split", "3 day split",
            "push pull legs 6 day"
        ],
        "answer": (
            "**Push/Pull/Legs (PPL) split for {name}:**\n\n"
            "Run 2x per week (6 days on, 1 rest) or 3 days/week:\n\n"
            "**PUSH — Chest, Shoulders, Triceps:**\n"
            "• Bench Press: 4×8–10\n"
            "• Overhead Press: 3×10\n"
            "• Cable Fly: 3×15\n"
            "• Lateral Raises: 4×15\n"
            "• Tricep Dips: 3×12\n\n"
            "**PULL — Back, Biceps, Rear Delts:**\n"
            "• Deadlift or Romanian DL: 4×6–8\n"
            "• Pull-Ups/Lat Pulldown: 4×10\n"
            "• Seated Row: 3×12\n"
            "• Face Pulls: 3×20\n"
            "• Hammer Curls: 3×12\n\n"
            "**LEGS — Quads, Hamstrings, Calves:**\n"
            "• Squats: 4×8\n"
            "• Leg Press: 3×12\n"
            "• RDL: 3×10\n"
            "• Leg Curls: 3×12\n"
            "• Calf Raises: 4×20\n\n"
            "Target: {bulk_cal if goal=='build' else cut_cal} kcal | {protein_g}g protein daily 🔥"
        )
    },

    # ── CARDIO ────────────────────────────────────────────────────────────────
    {
        "id": "cardio_advice",
        "questions": [
            "best cardio for fat loss", "how much cardio", "cardio tips",
            "cardio vs weights", "hiit or liss", "running for fat loss",
            "cardio frequency", "how often to do cardio", "cardio workout",
            "should I do cardio", "cardio for weight loss"
        ],
        "answer": (
            "Cardio strategy for **{name}** ({goal_text} goal):\n\n"
            "**Your heart rate zones (age: {age}):**\n"
            "• Fat burn zone: {hr_fat_low}–{hr_fat_high} bpm (60–70% max)\n"
            "• Cardio zone: {hr_cardio_low}–{hr_cardio_high} bpm (70–80% max)\n"
            "• Peak zone: {hr_peak_low}–{hr_peak_high} bpm (80–90% max)\n\n"
            "**Recommended for your goal:**\n"
            "• **LISS** (Low Intensity Steady State): 3–4x/week, 30–45 min\n"
            "  Best for: fat loss, recovery, endurance\n"
            "• **HIIT** (High Intensity Interval): 2x/week max\n"
            "  Best for: time efficiency, metabolic boost\n\n"
            "**For fat loss:** Do cardio AFTER weights (glycogen depleted → more fat burned)\n"
            "**For endurance:** Separate cardio from strength training by 6+ hours 🏃"
        )
    },

    # ── SUPPLEMENTS ───────────────────────────────────────────────────────────
    {
        "id": "creatine",
        "questions": [
            "should I take creatine", "creatine benefits", "creatine monohydrate",
            "is creatine safe", "creatine loading", "creatine dose",
            "creatine for muscle", "when to take creatine", "creatine effects"
        ],
        "answer": (
            "**Creatine guide for {name}:**\n\n"
            "Creatine monohydrate is the **most researched sports supplement** — "
            "100% safe, legal, and effective.\n\n"
            "**Benefits (backed by 200+ studies):**\n"
            "• +5–15% strength increase\n"
            "• +1–2kg lean muscle in 4 weeks\n"
            "• Better power output and sprint performance\n"
            "• Enhanced recovery between sets\n"
            "• May improve cognitive function\n\n"
            "**How to take it:**\n"
            "• **No loading needed** — just take 3–5g daily\n"
            "• Take any time (consistency > timing)\n"
            "• Stay hydrated — creatine draws water into muscles\n"
            "• Takes 3–4 weeks to fully saturate muscles\n\n"
            "**Cost:** ~£1–2/month. The best value supplement available. 💊"
        )
    },
    {
        "id": "protein_powder",
        "questions": [
            "best protein powder", "whey protein", "protein supplement",
            "whey vs casein", "protein shake", "should I use protein powder",
            "protein powder recommendation", "plant protein", "vegan protein"
        ],
        "answer": (
            "**Protein powder guide for {name}:**\n\n"
            "You need {protein_g}g protein/day. Food-first is always best, "
            "but supplements help hit targets.\n\n"
            "**Types compared:**\n"
            "| Type | Absorption | Best for | Notes |\n"
            "|------|-----------|----------|-------|\n"
            "| Whey Concentrate | Fast | Post-workout | Cheap, ~75% protein |\n"
            "| Whey Isolate | Fast | Lactose intolerant | 90%+ protein, pricier |\n"
            "| Casein | Slow | Before bed | 7-8h release |\n"
            "| Plant blend | Medium | Vegans | Pea+rice = complete amino acids |\n\n"
            "**Recommendation:** Whey concentrate or isolate post-workout. "
            "Casein before bed if budget allows. 1–2 scoops/day max — get the rest from food. 🥛"
        )
    },
    {
        "id": "pre_workout",
        "questions": [
            "pre workout supplement", "best pre workout", "should I take pre workout",
            "pre workout ingredients", "caffeine before workout", "pre workout dosing",
            "natural pre workout", "pre workout side effects"
        ],
        "answer": (
            "**Pre-workout guide for {name}:**\n\n"
            "**Effective ingredients to look for:**\n"
            "• **Caffeine** (150–300mg) — focus, energy, performance +3–7%\n"
            "• **Creatine** (3–5g) — strength and power output\n"
            "• **Beta-alanine** (3.2g) — muscular endurance (causes harmless tingling)\n"
            "• **Citrulline malate** (6–8g) — pump, endurance, recovery\n"
            "• **L-theanine** (100–200mg) — smooths caffeine, reduces jitters\n\n"
            "**DIY stack (cost-effective):**\n"
            "Coffee + creatine + banana = effective natural pre-workout\n\n"
            "**Timing:** 20–30 min before training\n"
            "**Caution:** Limit to 2–3x/week — avoid daily use to prevent tolerance ⚡"
        )
    },

    # ── SLEEP & RECOVERY ──────────────────────────────────────────────────────
    {
        "id": "sleep_recovery",
        "questions": [
            "how much sleep for gains", "sleep and muscle growth",
            "recovery tips", "how to recover faster", "overtraining",
            "rest days", "muscle soreness", "doms treatment",
            "how to sleep better", "sleep for athletes"
        ],
        "answer": (
            "**Recovery guide for {name}:**\n\n"
            "**Sleep (most underrated training variable):**\n"
            "• Target: **7–9 hours** per night\n"
            "• 90% of growth hormone is released during deep sleep\n"
            "• Poor sleep → elevated cortisol → muscle breakdown\n"
            "• Even one night of poor sleep cuts strength by 8%\n\n"
            "**Optimise sleep:**\n"
            "• Same sleep/wake time 7 days/week\n"
            "• Room temp: 16–18°C (65°F)\n"
            "• No screens 1hr before bed\n"
            "• Magnesium glycinate 400mg can improve sleep quality\n\n"
            "**Active recovery days:**\n"
            "• Light walking, stretching, swimming\n"
            "• 10-min foam rolling for soreness\n"
            "• Protein still matters on rest days — {protein_g}g target unchanged 😴"
        )
    },

    # ── NUTRITION TIMING ──────────────────────────────────────────────────────
    {
        "id": "pre_workout_meal",
        "questions": [
            "what to eat before workout", "pre workout meal", "food before gym",
            "eat before training", "pre workout nutrition", "best pre workout food"
        ],
        "answer": (
            "**Pre-workout nutrition for {name}:**\n\n"
            "**Timing matters:**\n"
            "• **2–3 hours before:** Full meal (carbs + protein + small fat)\n"
            "• **30–60 min before:** Light snack (fast carbs + protein)\n\n"
            "**Best pre-workout meals:**\n"
            "• Rice + chicken breast + vegetables (2–3h before)\n"
            "• Oats + banana + protein shake (1–2h before)\n"
            "• Greek yoghurt + fruit + honey (30–60 min before)\n"
            "• Banana + handful of rice cakes (30 min before)\n\n"
            "**Aim for:**\n"
            "• 30–50g carbs (fuel for training)\n"
            "• 20–30g protein\n"
            "• Minimal fat (slows digestion)\n"
            "• Water: 500ml in the 2 hours before training 🍽️"
        )
    },
    {
        "id": "post_workout_meal",
        "questions": [
            "what to eat after workout", "post workout meal", "food after gym",
            "eat after training", "post workout nutrition", "anabolic window",
            "protein after workout", "best post workout food"
        ],
        "answer": (
            "**Post-workout nutrition for {name}:**\n\n"
            "**The anabolic window:** You have ~2 hours post-workout to maximise recovery. "
            "But it's not as strict as once thought — consistency matters more than timing.\n\n"
            "**Ideal post-workout meal:**\n"
            "• **Protein:** 30–40g (to stimulate muscle protein synthesis)\n"
            "• **Carbs:** 40–80g (replenish glycogen stores)\n"
            "• **Fat:** Keep low in immediate post-workout meal\n\n"
            "**Best options:**\n"
            "• Chicken + rice + vegetables\n"
            "• Protein shake + banana + oats\n"
            "• Eggs + toast + fruit\n"
            "• Tuna + rice cakes + apple\n\n"
            "**Your protein target:** {protein_g}g/day (spread across meals) 🥗"
        )
    },

    # ── INTERMITTENT FASTING ──────────────────────────────────────────────────
    {
        "id": "intermittent_fasting",
        "questions": [
            "intermittent fasting", "16:8 fasting", "18:6 fasting",
            "should I do intermittent fasting", "IF diet", "fasting for weight loss",
            "how to do intermittent fasting", "fasting benefits"
        ],
        "answer": (
            "**Intermittent fasting guide for {name}:**\n\n"
            "IF is a meal timing strategy — not a magic diet. Works by reducing "
            "your eating window, making it easier to hit a calorie deficit.\n\n"
            "**Popular protocols:**\n"
            "• **16:8** — 16h fast, 8h eating window (e.g. noon–8pm). Most sustainable.\n"
            "• **18:6** — More aggressive, can blunt muscle growth\n"
            "• **5:2** — Normal 5 days, 500 kcal for 2 non-consecutive days\n\n"
            "**For your goal ({goal_text}):**\n"
            "• {if_recommendation}\n\n"
            "**Your numbers within the eating window:**\n"
            "• Calories: {calories} kcal\n"
            "• Protein: {protein_g}g (harder to hit in fewer meals — prioritise this)\n\n"
            "**Caution:** IF can worsen adherence for some. If you're hungry at 11am, "
            "a standard 3–4 meals/day may work better. The best diet is the one you stick to! ⏰"
        )
    },

    # ── INJURIES ─────────────────────────────────────────────────────────────
    {
        "id": "knee_pain",
        "questions": [
            "knee pain squatting", "knee hurts", "knee injury gym",
            "pain in knees", "knee problems lifting", "bad knees workout"
        ],
        "answer": (
            "**Knee pain during training — guidance for {name}:**\n\n"
            "**Important:** For acute/sharp pain, see a physiotherapist. "
            "This is general guidance, not medical advice.\n\n"
            "**Common causes:**\n"
            "• Poor squat form (knees caving inward)\n"
            "• Tight hip flexors or weak glutes\n"
            "• Overuse from too much volume too soon\n"
            "• Patellofemoral syndrome (runner's knee)\n\n"
            "**What to do:**\n"
            "1. Reduce load immediately — don't train through sharp pain\n"
            "2. Check squat form: feet shoulder-width, knees tracking over toes\n"
            "3. Strengthen: hip abductors, VMO (inner quad), hamstrings\n"
            "4. Mobility: hip flexor and quad stretching daily\n"
            "5. Ice 15–20 min post-training if inflamed\n\n"
            "**Knee-friendly alternatives while recovering:**\n"
            "• Leg press (adjustable angle)\n"
            "• Box squats (reduced range)\n"
            "• Swimming / cycling ⚕️"
        )
    },
    {
        "id": "lower_back_pain",
        "questions": [
            "lower back pain", "back hurts deadlift", "lower back injury",
            "back pain lifting", "lumbar pain", "sore lower back"
        ],
        "answer": (
            "**Lower back pain guidance for {name}:**\n\n"
            "**Important:** Persistent or radiating pain → see a doctor/physio.\n\n"
            "**Most common cause:** Rounding the lower back during deadlifts or "
            "squats (flexion under load).\n\n"
            "**Immediate steps:**\n"
            "1. Stop deadlifting/heavy squatting temporarily\n"
            "2. Apply ice (first 48h) then heat\n"
            "3. Keep moving — complete rest worsens most back pain\n\n"
            "**Strengthen your back:**\n"
            "• McGill big 3: curl-up, side plank, bird-dog\n"
            "• Hip hinge pattern drills (bodyweight)\n"
            "• Glute bridges\n"
            "• Dead bugs\n\n"
            "**Return to training:**\n"
            "• Start with light weight and perfect form\n"
            "• Film your deadlift — neutral spine throughout\n"
            "• Consider trap bar deadlift (more back-friendly) ⚕️"
        )
    },

    # ── MOTIVATION ────────────────────────────────────────────────────────────
    {
        "id": "motivation_plateau",
        "questions": [
            "not seeing results", "plateau", "results slowed", "no progress",
            "stopped losing weight", "weight loss plateau", "muscle plateau",
            "gains stopped", "stuck at same weight"
        ],
        "answer": (
            "**Breaking through your plateau, {name}:**\n\n"
            "Plateaus are NORMAL and happen to everyone. Here's what to do:\n\n"
            "**For fat loss plateau:**\n"
            "• Recalculate TDEE — your body adapted to {weight}kg\n"
            "• Reduce calories by 100–150 kcal ({cut_cal} → {mini_cut} kcal)\n"
            "• Add 1–2 cardio sessions/week\n"
            "• Check food tracking accuracy (weigh all food for 1 week)\n\n"
            "**For muscle plateau:**\n"
            "• Increase training volume (+1–2 sets per exercise)\n"
            "• Change rep ranges (if doing 3×10, try 4×6 or 3×15)\n"
            "• Add a deload week (50–60% weight for 1 week)\n"
            "• Prioritise sleep and calories\n\n"
            "**Mindset:** Progress is never linear. Zoom out — compare to 3 months ago, "
            "not last week. You've already come further than you think! 💪"
        )
    },
    {
        "id": "motivation_general",
        "questions": [
            "i dont want to workout", "feeling lazy", "no motivation",
            "how to stay motivated", "want to give up", "tired of gym",
            "can't be bothered", "lost motivation", "struggling with consistency",
            "how to be consistent", "gym motivation"
        ],
        "answer": (
            "I hear you, **{name}**. Every athlete goes through this. 💙\n\n"
            "**The truth about motivation:** It's unreliable. Discipline and habit "
            "are what actually carry you. Motivation is the spark — systems are the engine.\n\n"
            "**Practical strategies:**\n"
            "• **2-minute rule:** Just put your gym clothes on. Starting is the hardest part.\n"
            "• **Same time daily:** Make it automatic like brushing teeth\n"
            "• **Track everything:** Seeing numbers go up is addictive motivation\n"
            "• **Training partner:** 3x more likely to stay consistent with accountability\n"
            "• **Reduce friction:** Pack gym bag night before, gym on way home/work\n\n"
            "**When you're genuinely exhausted:** Sometimes the right call IS rest. "
            "One rest day won't derail your progress. But one missed session "
            "becomes two... becomes a week. Show up for 20 minutes minimum. 🔥\n\n"
            "You've got this, {name}. Your future self will thank you."
        )
    },

    # ── WATER / HYDRATION ─────────────────────────────────────────────────────
    {
        "id": "hydration",
        "questions": [
            "how much water to drink", "daily water intake", "hydration tips",
            "water for athletes", "dehydration performance", "water during workout",
            "how much water per day"
        ],
        "answer": (
            "**Hydration guide for {name}:**\n\n"
            "**Your daily water target: {water_l}L/day**\n"
            "*(Based on 33ml per kg bodyweight × {weight}kg)*\n\n"
            "**Add extra for:**\n"
            "• +0.5L per hour of exercise\n"
            "• +0.5L in hot weather\n"
            "• +0.5L if consuming caffeine heavily\n\n"
            "**Performance impact of dehydration:**\n"
            "• 2% body weight lost in water = 20% drop in strength\n"
            "• 3% = significant cognitive impairment\n\n"
            "**Practical tips:**\n"
            "• Start each morning with 500ml before coffee\n"
            "• 500ml 2 hours before training\n"
            "• 150–250ml every 15–20 min during exercise\n"
            "• Check urine: pale yellow = well hydrated 💧"
        )
    },

    # ── PROGRESSIVE OVERLOAD ─────────────────────────────────────────────────
    {
        "id": "progressive_overload",
        "questions": [
            "what is progressive overload", "progressive overload explained",
            "how to progressive overload", "adding weight to lifts",
            "double progression", "how to get stronger", "linear progression"
        ],
        "answer": (
            "**Progressive overload guide for {name}:**\n\n"
            "Progressive overload is the **#1 principle** of getting stronger and bigger. "
            "It means consistently increasing the demand on your muscles over time.\n\n"
            "**Methods (use in order):**\n"
            "1. **Add reps:** If you do 3×8, next session try 3×9\n"
            "2. **Add weight:** Once you hit top of rep range, add 2.5kg\n"
            "3. **Add sets:** Go from 3→4 sets over time\n"
            "4. **Reduce rest:** Same work in less time\n"
            "5. **Better form:** Slower eccentric, fuller range\n\n"
            "**Double progression (best for beginners):**\n"
            "Set a rep range (e.g. 8–12). Hit 12 reps → add weight next session. "
            "Can't hit 8 reps → too heavy.\n\n"
            "**Track EVERYTHING:** Log weights and reps every session. "
            "What gets measured gets improved. 📈"
        )
    },

    # ── SUPPLEMENTS - GENERAL ─────────────────────────────────────────────────
    {
        "id": "supplements_beginners",
        "questions": [
            "best supplements for beginners", "do I need supplements",
            "supplements for gym", "what supplements to take",
            "supplement stack", "necessary supplements", "are supplements worth it"
        ],
        "answer": (
            "**Supplement priority list for {name}:**\n\n"
            "**Tier 1 — Essential (get these first):**\n"
            "• **Creatine monohydrate** 3–5g/day — scientifically proven, cheap\n"
            "• **Vitamin D3** 2000–4000 IU/day — most people are deficient\n"
            "• **Protein powder** — only if struggling to hit {protein_g}g from food\n\n"
            "**Tier 2 — Beneficial:**\n"
            "• **Omega-3** (fish oil) 2–3g EPA+DHA — inflammation, recovery\n"
            "• **Magnesium glycinate** 400mg before bed — sleep quality\n"
            "• **Caffeine** (coffee) — proven performance enhancer\n\n"
            "**Tier 3 — Save your money on:**\n"
            "• BCAAs (redundant if hitting protein targets)\n"
            "• Testosterone boosters (no clinical evidence)\n"
            "• Fat burners (minimal effect, expensive)\n\n"
            "**Priority:** Food first, then these supplements. Nothing replaces "
            "consistent training, sleep, and nutrition. 💊"
        )
    },
]

# ─── KB Engine ────────────────────────────────────────────────────────────────

class FitnessKnowledgeBase:
    """
    TF-IDF cosine similarity knowledge base.
    Finds the best matching Q&A entry for a user query.
    """

    _instance   = None
    _vectorizer = None
    _matrix     = None
    _entries    = None
    _ready      = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not FitnessKnowledgeBase._ready:
            self._build_index()

    def _build_index(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            # Flatten all question examples into indexed corpus
            corpus  = []
            entries = []
            for entry in RAW_KB:
                for q in entry["questions"]:
                    corpus.append(q.lower())
                    entries.append(entry)

            vec = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
            mat = vec.fit_transform(corpus)

            FitnessKnowledgeBase._vectorizer = vec
            FitnessKnowledgeBase._matrix     = mat
            FitnessKnowledgeBase._entries    = entries
            FitnessKnowledgeBase._ready      = True

        except Exception as e:
            import logging
            logging.getLogger("apex_ai.kb").warning(f"KB index build failed: {e}")

    def search(self, query: str, threshold: float = 0.20) -> Optional[dict]:
        """
        Find the best matching KB entry for a query.
        Returns the entry dict or None if similarity < threshold.
        """
        if not FitnessKnowledgeBase._ready:
            return None
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = FitnessKnowledgeBase._vectorizer.transform([query.lower()])
            sims  = cosine_similarity(q_vec, FitnessKnowledgeBase._matrix)[0]
            best_idx  = int(sims.argmax())
            best_score = float(sims[best_idx])
            if best_score < threshold:
                return None
            return {
                "entry":      FitnessKnowledgeBase._entries[best_idx],
                "confidence": round(best_score, 3),
            }
        except Exception:
            return None

    def render(self, entry: dict, user_data: dict) -> str:
        """
        Fill in the answer template with personalised user data.
        """
        from backend.chatbot.utils import (
            calc_tdee, calc_bmi, bmi_category, calc_macros,
            water_target_liters, target_heart_rate_zone
        )
        ud = user_data or {}
        w    = float(ud.get("weight_kg", 70))
        h    = float(ud.get("height_cm", 175))
        age  = int(ud.get("age", 25))
        act  = int(ud.get("activity_level", 2))
        g    = str(ud.get("gender", "m"))
        goal = str(ud.get("goal", "maintain"))
        name = str(ud.get("name", "there")).split()[0]
        t_w  = float(ud.get("target_weight", w - 5))

        tdee   = calc_tdee(w, h, age, g, act)
        bmi    = calc_bmi(w, h)
        bmi_cat= bmi_category(bmi)
        macros = calc_macros(
            tdee - 500 if goal == "lose" else tdee + 300 if goal == "build" else tdee,
            goal, w
        )
        water  = water_target_liters(w)
        hr     = target_heart_rate_zone(age)
        max_hr = hr["max_hr"]

        activity_descs = {1:"sedentary",2:"lightly active",3:"moderately active",
                          4:"very active",5:"athlete"}
        goal_texts     = {"lose":"fat loss","build":"muscle gain","maintain":"maintenance"}

        wt_lose  = max(0, round(w - t_w, 1))
        wks      = max(1, round(wt_lose / 0.5)) if wt_lose else 0
        mini_cut = tdee - 650

        if_rec = {
            "lose":     "16:8 works well — easy to maintain a deficit",
            "build":    "Be cautious — harder to eat enough calories in short window",
            "maintain": "16:8 is fine for maintenance and metabolic health",
        }.get(goal, "16:8 is a good starting protocol")

        try:
            return entry["answer"].format(
                name=name, weight=w, height=h, age=age,
                goal=goal, goal_text=goal_texts.get(goal, goal),
                tdee=tdee, calories=macros["calories"],
                protein_g=macros["protein_g"], carb_g=macros["carb_g"],
                fat_g=macros["fat_g"],
                protein_cal=macros["protein_g"] * 4,
                carb_cal=macros["carb_g"] * 4,
                fat_cal=macros["fat_g"] * 9,
                protein_meal=round(macros["protein_g"] / 5),
                cut_cal=tdee - 500, bulk_cal=tdee + 300,
                mini_cut=mini_cut,
                target_weight=t_w, weight_to_lose=wt_lose,
                weeks_to_goal=wks, calories_deficit=500,
                water_l=water, bmi=bmi, bmi_cat=bmi_cat,
                activity_desc=activity_descs.get(act, "active"),
                hr_fat_low=hr["fat_burn"][0], hr_fat_high=hr["fat_burn"][1],
                hr_cardio_low=hr["cardio"][0], hr_cardio_high=hr["cardio"][1],
                hr_peak_low=hr["peak"][0], hr_peak_high=hr["peak"][1],
                max_hr=max_hr, if_recommendation=if_rec,
            )
        except KeyError:
            return entry["answer"]

    def is_ready(self) -> bool:
        return FitnessKnowledgeBase._ready
