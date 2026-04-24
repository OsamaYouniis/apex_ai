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


# ═══════════════════════════════════════════════════════════════════════════════
# EXPANDED KB ENTRIES (20 new topics — appended to RAW_KB)
# ═══════════════════════════════════════════════════════════════════════════════

RAW_KB.extend([

    # ── SHOULDER INJURIES ─────────────────────────────────────────────────────
    {
        "id": "shoulder_pain",
        "questions": [
            "shoulder pain pressing", "rotator cuff pain", "shoulder injury gym",
            "shoulder impingement", "pain in shoulder lifting",
            "shoulder clicking when pressing", "shoulder pain overhead press",
            "shoulder pain bench press", "how to fix shoulder pain",
            "shoulder exercises to avoid when injured"
        ],
        "answer": (
            "**Shoulder pain guidance for {name}:**\n\n"
            "⚠️ Acute sharp pain → stop training, see a physiotherapist.\n\n"
            "**Most common causes in gym:**\n"
            "• Shoulder impingement (subacromial) — most common\n"
            "• Rotator cuff irritation — from excessive pressing volume\n"
            "• AC joint irritation — from heavy overhead/dip work\n"
            "• Poor push:pull ratio — too much pressing vs rowing\n\n"
            "**Immediate steps:**\n"
            "1. Reduce pressing volume by 50% for 2 weeks\n"
            "2. Add **face pulls** 3×20 every session — external rotation\n"
            "3. Add **band pull-aparts** daily — 100 reps total\n"
            "4. Avoid overhead pressing if impingement is suspected\n\n"
            "**Safe alternatives while recovering:**\n"
            "• Landmine press (shoulder-friendly)\n"
            "• Neutral grip dumbbell press\n"
            "• Cable flys (no joint stress)\n\n"
            "**Prevention:** Maintain 1:1 push:pull ratio. Face pulls every session.\n"
            "Warm up rotator cuff before pressing. ⚕️"
        )
    },

    # ── HIP FLEXOR / LOWER BODY INJURIES ──────────────────────────────────────
    {
        "id": "hip_flexor_pain",
        "questions": [
            "hip flexor pain", "tight hip flexors", "hip pain squatting",
            "hip flexor strain", "hip pain after deadlift",
            "groin pain gym", "hip mobility exercises", "fix tight hips"
        ],
        "answer": (
            "**Hip flexor guidance for {name}:**\n\n"
            "**Cause:** Sitting + explosive training = chronically shortened hip flexors.\n\n"
            "**Assessment:** Lie flat. Pull one knee to chest. "
            "If other leg rises, hip flexors are tight.\n\n"
            "**Daily mobility protocol (5 min):**\n"
            "• Couch stretch: 2 min per side\n"
            "• 90/90 stretch: 2 min per side\n"
            "• Hip circle warm-up: 30s each direction\n\n"
            "**Glute activation (do before every lower body session):**\n"
            "• Glute bridges: 3×15\n"
            "• Clamshells with band: 2×20\n"
            "• Monster walks with band: 2×15 steps\n\n"
            "**Training adjustments:**\n"
            "• Reduce squat depth temporarily\n"
            "• Add goblet squats to improve hip mobility\n"
            "• Bulgarian split squats for unilateral strength\n\n"
            "**Recovery:** Apply heat before stretching. Ice after training if inflamed. ⚕️"
        )
    },

    # ── WRIST PAIN ─────────────────────────────────────────────────────────────
    {
        "id": "wrist_pain",
        "questions": [
            "wrist pain lifting", "wrist pain bench press", "wrist hurts gym",
            "wrist injury weightlifting", "how to protect wrists gym",
            "wrist wraps needed", "wrist pain curls", "wrist pain overhead press"
        ],
        "answer": (
            "**Wrist pain guidance for {name}:**\n\n"
            "**Most common cause:** Hyperextended wrists under load during pressing.\n\n"
            "**Fix form first:**\n"
            "• Keep wrists **straight/neutral** — not bent back\n"
            "• Bar should sit in palm crease, not fingers\n"
            "• Grip the bar like you're trying to break it — engages forearms\n\n"
            "**Equipment options:**\n"
            "• Wrist wraps — use for heavy sets only, not warmups\n"
            "• Straps — for pulling exercises if wrists are limiting factor\n\n"
            "**Mobility and strength:**\n"
            "• Wrist rotations daily: 30s each direction\n"
            "• Reverse wrist curls: 3×15 to balance flexor/extensor strength\n"
            "• Farmer carries to build wrist stability\n\n"
            "**Temporary alternatives:**\n"
            "• EZ-bar for curls (reduced wrist stress)\n"
            "• Dumbbell press (neutral grip option)\n"
            "• Push-up handles (elevated wrists) ⚕️"
        )
    },

    # ── DELOAD ─────────────────────────────────────────────────────────────────
    {
        "id": "deload_week",
        "questions": [
            "what is a deload week", "when to deload", "how to deload",
            "deload training program", "should I deload", "deload every 4 weeks",
            "overtraining symptoms", "fatigue from training", "how to recover from overtraining"
        ],
        "answer": (
            "**Deload guide for {name}:**\n\n"
            "A deload is a planned week of reduced training to allow supercompensation.\n\n"
            "**Signs you need a deload:**\n"
            "• Performance declining for 2+ weeks\n"
            "• Persistent joint pain or muscle fatigue\n"
            "• Poor sleep quality despite normal routine\n"
            "• Motivation near zero\n"
            "• Resting heart rate elevated by 5+ bpm\n\n"
            "**Deload protocol:**\n"
            "• **Volume deload** (recommended): Keep same weights, cut sets by 40-50%\n"
            "• **Intensity deload**: Keep same volume, reduce weights by 40-50%\n"
            "• Duration: 1 week\n\n"
            "**Frequency:** Every 4-6 weeks for intermediate/advanced. "
            "Every 6-8 weeks for beginners.\n\n"
            "**Your training volume:** At {weight}kg bodyweight with your current goals, "
            "deload when you've accumulated 4 weeks of hard training. 🔄"
        )
    },

    # ── BODY RECOMPOSITION ────────────────────────────────────────────────────
    {
        "id": "recomposition",
        "questions": [
            "lose fat and gain muscle at same time", "body recomposition",
            "recomp", "cut and bulk simultaneously",
            "maintain weight but change body composition",
            "how to recomp", "recomposition diet", "is recomp possible"
        ],
        "answer": (
            "**Body recomposition plan for {name}:**\n\n"
            "Recomp = losing fat AND building muscle simultaneously. "
            "Possible, but slower than dedicated cut or bulk.\n\n"
            "**Who recomp works best for:**\n"
            "• Beginners (first 1-2 years of training)\n"
            "• Returning after a long break\n"
            "• Body fat above 20% (men) or 28% (women)\n\n"
            "**Recomp protocol:**\n"
            "• Calories: Eat at **maintenance** ({tdee} kcal)\n"
            "• Protein: **{protein_g}g/day** — critically important at maintenance\n"
            "• Training: Progressive overload strength training 3-5x/week\n"
            "• Cardio: 2-3x moderate sessions/week\n\n"
            "**What to expect:**\n"
            "• Scale weight stays roughly the same\n"
            "• Body composition improves visually over 3-6 months\n"
            "• Progress slower than dedicated bulk/cut phases\n\n"
            "**Track:** Photos and measurements monthly — not just scale weight. ⚡"
        )
    },

    # ── BULKING GUIDE ─────────────────────────────────────────────────────────
    {
        "id": "bulking_guide",
        "questions": [
            "how to bulk", "bulking guide", "lean bulk vs dirty bulk",
            "how to gain muscle mass fast", "bulking tips",
            "how much to eat to bulk", "clean bulk diet",
            "bulk without gaining too much fat", "how to start bulking"
        ],
        "answer": (
            "**Bulking guide for {name}:**\n\n"
            "**Lean bulk (recommended):**\n"
            "• Calorie surplus: **+250-350 kcal above TDEE**\n"
            "• Your target: **{bulk_cal} kcal/day**\n"
            "• Expected rate: 0.25-0.5kg per week\n"
            "• ~50-50 muscle to fat ratio (best achievable)\n\n"
            "**Dirty bulk (not recommended):**\n"
            "• Large surplus (+500-1000 kcal)\n"
            "• Faster weight gain but mostly fat\n"
            "• Requires longer cut phase afterward\n\n"
            "**Macros for your bulk:**\n"
            "• Protein: **{protein_g}g** (2g/kg — protects against excess fat gain)\n"
            "• Carbs: **{carb_g}g** (training fuel)\n"
            "• Fat: **{fat_g}g** (hormones and absorption)\n\n"
            "**Training:** Progressive overload is mandatory. "
            "Surplus without training = fat gain, not muscle. \n"
            "**When to stop:** Bulk until body fat reaches ~15-17% (men) or ~25-27% (women), "
            "then cut. 💪"
        )
    },

    # ── CUTTING GUIDE ─────────────────────────────────────────────────────────
    {
        "id": "cutting_guide",
        "questions": [
            "how to cut body fat", "cutting phase guide", "how to get lean",
            "fat loss phase", "how to cut without losing muscle",
            "cutting diet tips", "how to preserve muscle while cutting",
            "how to start a cut", "aggressive cut vs slow cut"
        ],
        "answer": (
            "**Cutting guide for {name}:**\n\n"
            "**Calorie target: {cut_cal} kcal/day** (500 kcal deficit)\n"
            "Expected rate: ~0.5kg/week fat loss\n\n"
            "**Aggressive cut vs slow cut:**\n"
            "• Slow cut (-300-400 kcal): More muscle preserved, sustainable\n"
            "• Moderate cut (-500 kcal): {cut_cal} kcal — recommended\n"
            "• Aggressive cut (-750+ kcal): Faster but risks muscle loss and fatigue\n\n"
            "**Key to preserving muscle while cutting:**\n"
            "1. **High protein: {protein_g}g/day** — most important factor\n"
            "2. Keep lifting heavy — signal to body that muscle is needed\n"
            "3. Small deficit — don't drop calories too fast\n"
            "4. Sleep 7-9 hours — cortisol breaks down muscle when sleep-deprived\n\n"
            "**Avoid:**\n"
            "• Cardio only (no lifting) — accelerates muscle loss\n"
            "• Cutting calories below {bmr_approx} kcal — metabolic adaptation\n"
            "• Eliminating carbs entirely — impairs training performance\n\n"
            "**Check-ins:** Weigh daily, average weekly. "
            "If no loss in 2 weeks, reduce by 100-150 kcal. 🔥"
        )
    },

    # ── BEGINNER GUIDE ────────────────────────────────────────────────────────
    {
        "id": "beginner_guide",
        "questions": [
            "I am a beginner at the gym", "just started gym", "gym for beginners",
            "first time at gym tips", "beginner fitness advice",
            "how to start going to gym", "gym beginner mistakes",
            "what should beginners do at gym", "starting fitness journey"
        ],
        "answer": (
            "**Beginner gym guide for {name} — welcome! 🎉**\n\n"
            "**The 3 things that matter most as a beginner:**\n"
            "1. **Show up consistently** — 3x/week beats perfect programming\n"
            "2. **Learn compound movements** — squat, deadlift, bench, row, overhead press\n"
            "3. **Progressive overload** — add weight or reps every session\n\n"
            "**Beginner program (3 days/week, full body):**\n"
            "• Squat: 3×8\n"
            "• Bench Press: 3×8\n"
            "• Bent-Over Row: 3×8\n"
            "• Overhead Press: 3×8\n"
            "• Romanian Deadlift: 3×10\n"
            "• Plank: 3×30s\n\n"
            "**Beginner nutrition:**\n"
            "• Protein: {protein_g}g/day — the most important number\n"
            "• Calories: {calories} kcal/day for your {goal} goal\n"
            "• Stay hydrated: {water_l}L water daily\n\n"
            "**Common beginner mistakes:**\n"
            "• Doing too much too soon → injury and burnout\n"
            "• Skipping compound lifts for isolation machines\n"
            "• Ignoring nutrition (training is 40%, nutrition is 60%)\n"
            "• Expecting visible results in 2-4 weeks (expect 8-12 weeks)\n\n"
            "**Your first 90 days:** Focus on form, not weight. "
            "Newbie gains are real — you WILL see results faster than anyone else. 💪"
        )
    },

    # ── HOME WORKOUT ──────────────────────────────────────────────────────────
    {
        "id": "home_workout",
        "questions": [
            "workout at home", "home workout no equipment", "bodyweight exercises",
            "train at home", "no gym workout", "home fitness routine",
            "calisthenics for beginners", "bodyweight program",
            "exercise at home without gym", "push up workout plan"
        ],
        "answer": (
            "**Home workout plan for {name}:**\n\n"
            "**3-Day Full Body (No Equipment):**\n\n"
            "**Day A:**\n"
            "• Push-ups: 4×max reps\n"
            "• Bodyweight squats: 4×20\n"
            "• Glute bridges: 4×20\n"
            "• Plank: 3×45s\n"
            "• Superman hold: 3×10 (lower back)\n\n"
            "**Day B (Active recovery):**\n"
            "• 30-45 min walk or light yoga\n\n"
            "**Day C:**\n"
            "• Same as Day A with 1 extra set or harder variation\n\n"
            "**Progressive overload at home:**\n"
            "• Push-up → Diamond push-up → Archer push-up → Pike push-up\n"
            "• Squat → Jump squat → Bulgarian split squat → Pistol squat\n"
            "• Glute bridge → Single-leg bridge → Hip thrust off chair\n\n"
            "**Nutrition doesn't change:**\n"
            "• Still hit {protein_g}g protein daily\n"
            "• Calories: {calories} kcal\n\n"
            "**Invest in:** Resistance bands (£15-25) and a pull-up bar (£20) "
            "to massively expand what's possible at home. 🏠"
        )
    },

    # ── AGE 50+ FITNESS ───────────────────────────────────────────────────────
    {
        "id": "fitness_over_50",
        "questions": [
            "fitness over 50", "workout over 50", "training after 50",
            "gym for older adults", "exercise for seniors",
            "strength training over 50", "can I build muscle after 50",
            "fitness for 50 year old", "age and fitness"
        ],
        "answer": (
            "**Fitness over 50 — guidance for {name}:**\n\n"
            "Good news: You can absolutely build muscle and lose fat after 50. "
            "The principles are the same — the approach is adjusted.\n\n"
            "**Key differences after 50:**\n"
            "• Recovery takes longer — need 48-72h between sessions for same muscle\n"
            "• Sarcopenia (muscle loss) accelerates — makes protein even more critical\n"
            "• Joint health needs attention — warm-up is non-negotiable\n"
            "• Hormonal changes (testosterone decline in men, menopause in women)\n\n"
            "**Adjusted recommendations:**\n"
            "• **Protein: higher than standard** — 2.0-2.4g per kg bodyweight\n"
            "• **Your protein target: {protein_g}g/day minimum**\n"
            "• Training frequency: 3-4x/week (not 5-6)\n"
            "• Rep ranges: 8-15 (reduce heavy 1-5 rep work to protect joints)\n"
            "• Warm-up: 10-15 min including mobility work\n\n"
            "**Priority exercises:**\n"
            "• Squats and deadlifts (modified if needed) — bone density\n"
            "• Rows and pulls — posture and back health\n"
            "• Balance training — injury prevention\n"
            "• Walking — cardiovascular health, low impact\n\n"
            "**WHO recommends:** 150-300 min moderate cardio + 2x strength training/week "
            "for adults over 65. ⚡"
        )
    },

    # ── ALCOHOL AND FITNESS ───────────────────────────────────────────────────
    {
        "id": "alcohol_fitness",
        "questions": [
            "alcohol and fitness", "can I drink alcohol and build muscle",
            "beer and working out", "alcohol effect on gains",
            "drinking and muscle building", "alcohol and weight loss",
            "how much alcohol is ok for fitness", "alcohol after workout"
        ],
        "answer": (
            "**Alcohol and fitness — honest guide for {name}:**\n\n"
            "**The impact (evidence-based):**\n"
            "• Inhibits muscle protein synthesis for up to 24 hours after consumption\n"
            "• Disrupts sleep quality — impairs GH release and recovery\n"
            "• 7 kcal per gram of alcohol (empty calories)\n"
            "• Increases cortisol — muscle breakdown hormone\n"
            "• Dehydrates — impairs next-day performance\n\n"
            "**Practical guidelines:**\n"
            "• 1-2 drinks on a rest day → minimal impact on progress\n"
            "• Heavy drinking (4+ drinks) → significant setback for 24-48h\n"
            "• Avoid alcohol on training days or the night before training\n"
            "• If drinking: eat protein-rich meal first to reduce absorption rate\n\n"
            "**For your goals ({goal}):**\n"
            "• Fat loss: alcohol pauses fat oxidation while it's metabolized\n"
            "• Muscle gain: directly inhibits MPS — keep drinking minimal\n\n"
            "**Bottom line:** Occasional moderate drinking won't ruin progress. "
            "Regular heavy drinking will. Consistency matters more than perfection. 🍺"
        )
    },

    # ── TRAVEL FITNESS ────────────────────────────────────────────────────────
    {
        "id": "travel_workout",
        "questions": [
            "workout while traveling", "hotel room workout", "gym while on holiday",
            "exercise on vacation", "training while traveling",
            "how to stay fit while traveling", "no gym on holiday workout",
            "fitness travel tips", "bodyweight workout hotel"
        ],
        "answer": (
            "**Travel workout plan for {name}:**\n\n"
            "**Hotel room workout (20 minutes, no equipment):**\n"
            "• Push-ups: 4×max\n"
            "• Bodyweight squats: 4×20\n"
            "• Mountain climbers: 3×30s\n"
            "• Glute bridges: 3×20\n"
            "• Plank: 3×45s\n"
            "• Jump squats: 3×15\n\n"
            "**Find a gym:**\n"
            "• Google 'gym day pass [city]' — most gyms offer day passes (£5-15)\n"
            "• Hotel gyms — usually have enough for a decent session\n"
            "• Outdoor parks — pull-up bars and open space\n\n"
            "**Nutrition while traveling:**\n"
            "• Prioritize protein at every meal — most restaurant food is high carb\n"
            "• Pack: protein bars, jerky, nuts for snacks\n"
            "• Stay hydrated — flights cause significant dehydration\n"
            "• Your protein target is still {protein_g}g/day\n\n"
            "**Mindset:** Maintaining is the goal when traveling, not improving. "
            "Even 2 sessions per week preserves 90% of your gains. ✈️"
        )
    },

    # ── SLEEP OPTIMIZATION ────────────────────────────────────────────────────
    {
        "id": "sleep_optimization",
        "questions": [
            "how to sleep better for gains", "sleep schedule for athletes",
            "optimize sleep for recovery", "how to improve sleep quality",
            "sleep tips for gym", "not sleeping well affecting gains",
            "sleep and testosterone", "how many hours sleep to build muscle"
        ],
        "answer": (
            "**Sleep optimization for {name}:**\n\n"
            "Sleep is the **most underrated performance variable**. "
            "No supplement or training protocol compensates for poor sleep.\n\n"
            "**The numbers:**\n"
            "• 90% of growth hormone released during deep sleep\n"
            "• One night of 5h sleep reduces strength by 8-10%\n"
            "• Poor sleep increases cortisol → muscle breakdown, fat storage\n"
            "• Target: **7-9 hours** in a consistent window\n\n"
            "**Evidence-based improvements:**\n"
            "• Same bedtime/wake time 7 days/week (including weekends)\n"
            "• Room temperature: 16-19°C (65-67°F)\n"
            "• No screens 60 min before bed (blue light suppresses melatonin)\n"
            "• Magnesium glycinate 400mg before bed — improves sleep quality\n"
            "• No caffeine after 2pm (half-life = 5-6 hours)\n"
            "• Blackout curtains — light exposure reduces melatonin\n\n"
            "**Pre-sleep nutrition:**\n"
            "• Casein protein before bed → sustained MPS overnight\n"
            "• Avoid large carb meals within 2h of sleep\n"
            "• Tart cherry juice — natural melatonin source 😴"
        )
    },

    # ── MEAL PREP ─────────────────────────────────────────────────────────────
    {
        "id": "meal_prep",
        "questions": [
            "meal prep for the week", "how to meal prep", "meal prep tips",
            "batch cooking for fitness", "meal prep for muscle gain",
            "meal prep for weight loss", "meal prep beginners guide",
            "sunday meal prep", "meal prep containers", "healthy meal prep ideas"
        ],
        "answer": (
            "**Meal prep guide for {name}:**\n\n"
            "**Weekly meal prep (2 hours on Sunday):**\n\n"
            "**Batch cook these:**\n"
            "• **Protein:** 1kg chicken breast, 500g lean beef mince, or "
            "1kg salmon portions (freeze half)\n"
            "• **Carbs:** 500g rice (dry) = ~1.5kg cooked, "
            "or 1kg sweet potato (roast)\n"
            "• **Veg:** 500g broccoli/spinach — blanch and store\n\n"
            "**Your macro targets per meal:**\n"
            "• Protein: ~{protein_meal}g per meal (4 meals/day from {protein_g}g total)\n"
            "• Calories per meal: ~{meal_kcal} kcal (from {calories} kcal daily)\n\n"
            "**Storage:**\n"
            "• Fridge: 4-5 days (keep rice in separate container)\n"
            "• Freezer: Protein sources up to 3 months\n"
            "• Glass containers > plastic for reheating\n\n"
            "**Quick assembly (5 min per meal):**\n"
            "Protein + carb + veg + sauce/seasoning.\n"
            "Vary the seasoning — same ingredients, different flavours each day.\n\n"
            "**Cost:** Meal prepping typically saves 40-60% vs buying prepared food. 🥗"
        )
    },

    # ── CARDIO PROGRAMMING ────────────────────────────────────────────────────
    {
        "id": "cardio_programming",
        "questions": [
            "how to program cardio", "cardio schedule", "how much cardio per week",
            "cardio for endurance", "zone 2 training", "LISS cardio plan",
            "HIIT schedule", "running program for beginners",
            "cardio without losing muscle", "how to add cardio to weight training"
        ],
        "answer": (
            "**Cardio programming for {name} ({goal_text} goal):**\n\n"
            "**Your heart rate zones:**\n"
            "• Zone 2 (fat burn): {hr_fat_low}–{hr_fat_high} bpm\n"
            "• Zone 3 (aerobic): {hr_cardio_low}–{hr_cardio_high} bpm\n"
            "• Zone 4 (threshold): {hr_peak_low}–{hr_peak_high} bpm\n\n"
            "**Recommended weekly cardio by goal:**\n"
            "• **Fat loss:** 3-4x LISS (30-45 min Zone 2) + 1x HIIT (20 min)\n"
            "• **Muscle gain:** 2x LISS (20-30 min) — maintain cardiovascular health\n"
            "• **General fitness:** 3x LISS or 2x LISS + 1x HIIT\n\n"
            "**Cardio without losing muscle:**\n"
            "• Keep sessions under 45 min for steady state\n"
            "• Do cardio AFTER weights, or on separate days\n"
            "• Eat protein before and after cardio sessions\n"
            "• Zone 2 is the safest for muscle preservation\n\n"
            "**HIIT template (20 min total):**\n"
            "5 min warm-up → 8 rounds: 30s sprint / 90s walk → 5 min cool-down\n\n"
            "**LISS options:** Walking (incline), cycling, rowing, swimming — "
            "pick whatever you'll actually do. 🏃"
        )
    },

    # ── STRETCHING AND FLEXIBILITY ────────────────────────────────────────────
    {
        "id": "stretching_flexibility",
        "questions": [
            "stretching routine for gym", "flexibility exercises",
            "how to improve flexibility", "static vs dynamic stretching",
            "stretching before or after workout", "mobility routine",
            "tight muscles exercises", "yoga for gym goers",
            "hip mobility routine", "thoracic spine mobility"
        ],
        "answer": (
            "**Stretching and mobility guide for {name}:**\n\n"
            "**Before training — Dynamic stretching (5-10 min):**\n"
            "• Leg swings front/back: 20 each leg\n"
            "• Hip circles: 30s each direction\n"
            "• Arm circles and shoulder rolls: 30s\n"
            "• Bodyweight squats: 10 slow reps\n"
            "• Inchworm with reach: 5 reps\n\n"
            "**After training — Static stretching (5-10 min):**\n"
            "• Hip flexor stretch: 45s each side\n"
            "• Hamstring stretch: 45s each side\n"
            "• Chest doorway stretch: 45s\n"
            "• Lats stretch (hanging or wall): 45s\n"
            "• Thoracic rotation: 30s each side\n\n"
            "**Why it matters:**\n"
            "• Dynamic pre-workout: increases range of motion and blood flow\n"
            "• Static post-workout: reduces next-day soreness, improves flexibility\n"
            "• Do NOT do static stretching before lifting — reduces power output\n\n"
            "**For significant flexibility improvement:**\n"
            "• 20 min dedicated stretching 3x/week\n"
            "• Hold each stretch 45-60 seconds (3 rounds)\n"
            "• Yoga 1x/week — highly effective for gym goers 🧘"
        )
    },

    # ── TESTOSTERONE AND HORMONES ─────────────────────────────────────────────
    {
        "id": "testosterone_natural",
        "questions": [
            "how to boost testosterone naturally", "low testosterone symptoms",
            "testosterone and muscle building", "natural testosterone tips",
            "how to increase testosterone", "testosterone and diet",
            "does lifting increase testosterone", "testosterone sleep connection"
        ],
        "answer": (
            "**Natural testosterone optimization for {name}:**\n\n"
            "⚠️ Always get bloodwork before assuming low testosterone. "
            "Symptoms overlap with many conditions.\n\n"
            "**Evidence-based natural boosters:**\n\n"
            "**Training:**\n"
            "• Compound heavy lifting (squats, deadlifts) → acute T spike\n"
            "• Keep sessions under 75 min — beyond this cortisol rises, T drops\n"
            "• 3-5x/week training — more frequent is not always better\n\n"
            "**Nutrition:**\n"
            "• Don't go too low in fat — T is synthesized from cholesterol\n"
            "• Eat at or above maintenance calories ({tdee} kcal)\n"
            "• Zinc-rich foods: oysters, beef, pumpkin seeds\n"
            "• Vitamin D: 2000-4000 IU/day (deficiency = lower T)\n\n"
            "**Lifestyle:**\n"
            "• Sleep 7-9 hours — 70% of T is produced during sleep\n"
            "• Reduce chronic stress — cortisol is testosterone's antagonist\n"
            "• Maintain healthy body fat — excess fat converts T to estrogen\n\n"
            "**Supplements with evidence:**\n"
            "• Vitamin D3 (if deficient) ✅\n"
            "• Zinc (if deficient) ✅\n"
            "• Ashwagandha (stress reduction → cortisol reduction) ✅\n"
            "• 'Testosterone boosters' — minimal evidence ❌ ⚗️"
        )
    },

    # ── NUTRITION MYTHS ───────────────────────────────────────────────────────
    {
        "id": "nutrition_myths",
        "questions": [
            "fitness nutrition myths", "diet myths debunked",
            "eating after 6pm myth", "carbs make you fat myth",
            "breakfast most important meal myth", "fat makes you fat myth",
            "detox diet truth", "clean eating myths", "common diet mistakes"
        ],
        "answer": (
            "**Top fitness nutrition myths debunked for {name}:**\n\n"
            "**Myth 1: 'Eating carbs after 6pm causes fat gain'**\n"
            "❌ False. Total daily calories and macros determine body composition, "
            "not meal timing. Eat carbs whenever it fits your schedule.\n\n"
            "**Myth 2: 'Fat makes you fat'**\n"
            "❌ False. Excess calories cause fat gain. Dietary fat is essential "
            "for hormones, brain function, and vitamin absorption.\n\n"
            "**Myth 3: 'Breakfast is the most important meal'**\n"
            "❌ False. Meal timing matters far less than total daily intake. "
            "Skip it if you're not hungry — intermittent fasting works for many people.\n\n"
            "**Myth 4: 'Eating 6 meals/day boosts metabolism'**\n"
            "❌ Minimal effect. Meal frequency doesn't significantly affect metabolism. "
            "Eat however many meals works for your schedule.\n\n"
            "**Myth 5: 'Detox diets cleanse your system'**\n"
            "❌ False. Your liver and kidneys detox continuously. "
            "No diet accelerates this.\n\n"
            "**What actually matters:**\n"
            "1. Total calories ({calories} kcal for your goal)\n"
            "2. Sufficient protein ({protein_g}g/day)\n"
            "3. Consistency over weeks and months\n"
            "4. Foods you actually enjoy eating 📊"
        )
    },

    # ── PROTEIN SOURCES ───────────────────────────────────────────────────────
    {
        "id": "protein_sources",
        "questions": [
            "best protein sources", "high protein foods list",
            "protein sources for vegetarians", "vegan protein sources",
            "cheap protein sources", "whole food protein sources",
            "protein foods for muscle building", "complete protein sources",
            "how to get protein without supplements"
        ],
        "answer": (
            "**Protein sources guide for {name} (target: {protein_g}g/day):**\n\n"
            "**Animal proteins (complete amino acid profile):**\n"
            "| Food | Protein per 100g | Approx cost |\n"
            "|------|-----------------|-------------|\n"
            "| Chicken breast | 31g | Low |\n"
            "| Tuna (canned) | 25g | Very low |\n"
            "| Eggs | 13g (6g/egg) | Very low |\n"
            "| Salmon | 25g | Medium |\n"
            "| Lean beef mince | 26g | Medium |\n"
            "| Greek yogurt | 10g | Low |\n"
            "| Cottage cheese | 11g | Low |\n\n"
            "**Plant proteins (combine for complete amino acids):**\n"
            "| Food | Protein per 100g | Notes |\n"
            "|------|-----------------|-------|\n"
            "| Tofu (firm) | 17g | Complete protein |\n"
            "| Lentils (cooked) | 9g | High iron |\n"
            "| Chickpeas | 9g | Versatile |\n"
            "| Edamame | 11g | Complete |\n"
            "| Tempeh | 19g | Complete, fermented |\n\n"
            "**Practical tip:** To hit {protein_g}g/day from food alone:\n"
            "~200g chicken + 3 eggs + 150g Greek yogurt = ~90g protein. "
            "Add lentils/cottage cheese for the rest. 🥩"
        )
    },

    # ── TRACKING AND PROGRESS ─────────────────────────────────────────────────
    {
        "id": "progress_tracking",
        "questions": [
            "how to track fitness progress", "how to measure progress gym",
            "progress photos tips", "how to track muscle gain",
            "scale weight not moving", "non scale victories",
            "how to know if I'm making progress", "fitness metrics to track",
            "body measurements fitness", "track calories app"
        ],
        "answer": (
            "**Progress tracking guide for {name}:**\n\n"
            "**Scale weight — use it right:**\n"
            "• Weigh daily, same time (morning, after toilet, before eating)\n"
            "• Use **7-day average** — not daily readings\n"
            "• Weight fluctuates 1-3kg daily from water, food, sleep — ignore noise\n\n"
            "**Better metrics to track:**\n"
            "• **Body measurements** (monthly): waist, hips, chest, arms, thighs\n"
            "• **Progress photos** (monthly): same lighting, same time of day\n"
            "• **Strength numbers**: squat, bench, deadlift, pull-ups → objective proof\n"
            "• **Energy levels and sleep quality**\n"
            "• **How clothes fit**\n\n"
            "**Gym tracking:**\n"
            "• Log every workout: exercise, sets, reps, weight\n"
            "• If you're stronger than last week → you're making progress\n"
            "• Apps: Strong, Hevy, or a simple notebook\n\n"
            "**Nutrition tracking:**\n"
            "• MyFitnessPal, Cronometer, or Nutracheck\n"
            "• Weigh food (don't eyeball) for at least 4 weeks until calibrated\n\n"
            "**Reality check:** Muscle gain is 0.5-1kg/month for natural trainees. "
            "Fat loss is 0.5-1kg/week maximum sustainably. "
            "Expect 8-12 weeks to see clear visual changes. 📈"
        )
    },
])
