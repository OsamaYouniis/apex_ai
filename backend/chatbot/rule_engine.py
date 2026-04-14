"""
backend/chatbot/rule_engine.py
────────────────────────────────
Advanced Rule-Based Response Engine — 50+ fitness topics.

This engine is the FALLBACK and also handles FACTUAL fitness questions
that are better served by deterministic, personalised formulas than a
language model (e.g. "how many calories do I need?" → we compute it precisely).

Every response is:
  - Personalised with the user's exact stats
  - Formatted with markdown bold and emojis
  - Medically accurate (uses Mifflin-St Jeor, WHO BMI, etc.)
"""

from backend.chatbot.utils import (
    calc_tdee, calc_bmi, bmi_category, calc_macros,
    water_target_liters, target_heart_rate_zone
)


def get_response(text: str, user_data: dict) -> str:
    """
    Main entry point.
    Returns a personalised, formatted fitness response string.
    """
    t    = text.lower().strip()
    ud   = user_data or {}

    # ── Extract user stats with safe defaults ─────────────────────────────
    name   = str(ud.get("name", "there")).split()[0]
    w      = float(ud.get("weight_kg", 70))
    h      = float(ud.get("height_cm", 175))
    age    = int(ud.get("age", 25))
    act    = int(ud.get("activity_level", 2))
    gender = str(ud.get("gender", "m"))
    goal   = str(ud.get("goal", "maintain"))
    t_w    = float(ud.get("target_weight", w - 5))

    tdee   = calc_tdee(w, h, age, gender, act)
    bmi    = calc_bmi(w, h)
    bmi_cat= bmi_category(bmi)
    macros = calc_macros(
        tdee - 500 if goal == "lose" else tdee + 300 if goal == "build" else tdee,
        goal, w
    )
    water  = water_target_liters(w)
    hr     = target_heart_rate_zone(age)
    protein= macros["protein_g"]
    cals   = macros["calories"]
    weeks_to_goal = round(abs(w - t_w) / 0.5) if w != t_w else 0

    # ─────────────────────────────────────────────────────────────────────
    # TOPIC HANDLERS — ordered by specificity
    # ─────────────────────────────────────────────────────────────────────

    # ── GREETINGS ─────────────────────────────────────────────────────────
    if _match(t, ["hello","hi ","hey ","good morning","good evening","good afternoon",
                  "sup ","what's up","salam","marhaba","howdy","greetings","yo "]):
        return (
            f"Hey {name}! 👋 Great to see you!\n\n"
            f"**Your Current Stats:**\n"
            f"• Weight: **{w}kg** | Height: **{h}cm** | Age: **{age}**\n"
            f"• BMI: **{bmi}** — {bmi_cat}\n"
            f"• Daily Calorie Target: **{cals} kcal**\n"
            f"• Protein Target: **{protein}g/day**\n"
            f"• Goal: **{goal.capitalize()} weight** → {t_w}kg\n\n"
            f"I'm APEX AI, your personal coach. Ask me anything:\n"
            f"💪 Workouts · 🥗 Nutrition · 💊 Supplements · 🔥 Fat loss · 📊 Stats"
        )

    # ── WORKOUT PLANS ────────────────────────────────────────────────────
    if _match(t, ["workout plan","training plan","weekly plan","ppl","push pull leg",
                  "5 day","4 day","6 day","full body","upper lower","exercise plan"]):
        plan_note = {
            "lose":     ("Fat Loss + Conditioning", "45–60s rest to maximise calorie burn"),
            "build":    ("Muscle Hypertrophy",       "90–120s rest, progressive overload each week"),
            "maintain": ("Balanced Strength",         "60–90s rest, focus on consistency"),
        }.get(goal, ("General Fitness", "60–90s rest"))

        return (
            f"**Your Personalised {plan_note[0]} Plan** 💪\n"
            f"*{name} · {w}kg · {h}cm · Goal: {goal} · {cals} kcal/day*\n\n"
            f"**Day 1 — Push (Chest · Shoulders · Triceps)**\n"
            f"• Flat Bench Press 4×10 @ 70% 1RM\n"
            f"• Incline Dumbbell Press 3×12\n"
            f"• Overhead Press 3×10\n"
            f"• Lateral Raises 4×15\n"
            f"• Tricep Rope Pushdown 3×15\n\n"
            f"**Day 2 — Pull (Back · Biceps · Rear Delts)**\n"
            f"• Deadlifts 4×6\n"
            f"• Pull-ups / Lat Pulldown 4×8\n"
            f"• Barbell Rows 3×10\n"
            f"• Face Pulls 3×15\n"
            f"• Barbell Curl 3×12\n\n"
            f"**Day 3 — Legs (Quads · Hamstrings · Glutes)**\n"
            f"• Back Squats 4×8\n"
            f"• Romanian Deadlift 3×10\n"
            f"• Leg Press 4×12\n"
            f"• Walking Lunges 3×12 each\n"
            f"• Calf Raises 4×20\n\n"
            f"**Day 4 — Rest / Active Recovery** 🧘\n\n"
            f"**Day 5 — Upper Strength**\n"
            f"• Weighted Pull-ups 4×6\n"
            f"• Bench Press 4×6 (heavy)\n"
            f"• Cable Rows 3×12\n"
            f"• Arnold Press 3×12\n\n"
            f"**Day 6 — Cardio + Core**\n"
            f"• 30 min HIIT or 45 min walk\n"
            f"• Plank 3×60s | Leg Raises 3×15\n\n"
            f"**Day 7 — Rest** 💤\n\n"
            f"**Key tip:** {plan_note[1]}\n"
            f"**Your daily protein:** {protein}g | **Calories:** {cals} kcal"
        )

    # ── HOME WORKOUT ─────────────────────────────────────────────────────
    if _match(t, ["home","no gym","no equipment","bodyweight","calisthenics"]):
        return (
            f"**Home Workout Plan — Zero Equipment** 🏠\n"
            f"*{name} · {w}kg · Goal: {goal}*\n\n"
            f"**Day 1 — Upper Push**\n"
            f"• Push-ups 4×15 (wide / close / diamond)\n"
            f"• Pike Push-ups 3×12 (shoulders)\n"
            f"• Tricep Dips on chair 3×15\n"
            f"• Superman holds 3×10\n\n"
            f"**Day 2 — Lower Body**\n"
            f"• Squats 4×20 | Jump Squats 3×10\n"
            f"• Reverse Lunges 3×12 each\n"
            f"• Glute Bridges 4×20\n"
            f"• Wall Sit 3×45s\n\n"
            f"**Day 3 — Core + Cardio**\n"
            f"• Burpees 3×10 | Mountain Climbers 3×30s\n"
            f"• Plank 3×60s | Bicycle Crunches 3×20\n"
            f"• High Knees 3×30s\n\n"
            f"**Day 4 — Rest | Repeat Days 1–3**\n\n"
            f"💡 *Progressive overload at home: slow the tempo, add pause reps, or wear a backpack with books!*"
        )

    # ── NUTRITION / MEAL PLAN ────────────────────────────────────────────
    if _match(t, ["diet","nutrition","meal plan","what to eat","food","eating","meal prep","macros"]):
        return (
            f"**Your Personalised Nutrition Plan** 🥗\n"
            f"*{name} · Goal: {goal} · {cals} kcal/day*\n\n"
            f"**Daily Macro Targets:**\n"
            f"• 🥩 Protein: **{macros['protein_g']}g** ({macros['protein_g']*4} kcal)\n"
            f"• 🍚 Carbs:   **{macros['carb_g']}g**   ({macros['carb_g']*4} kcal)\n"
            f"• 🥑 Fats:    **{macros['fat_g']}g**    ({macros['fat_g']*9} kcal)\n"
            f"• 🔥 Total:   **{cals} kcal**\n\n"
            f"**Sample Full-Day Plan:**\n\n"
            f"🌅 **Breakfast** (~520 kcal)\n"
            f"• 3 eggs + 80g oats + banana + black coffee\n\n"
            f"☀️ **Lunch** (~650 kcal)\n"
            f"• 200g chicken breast + 150g brown rice + salad\n\n"
            f"🍎 **Pre-Workout Snack** (~300 kcal)\n"
            f"• 1 scoop whey protein + 1 banana + almonds\n\n"
            f"🌙 **Dinner** (~580 kcal)\n"
            f"• 180g salmon + sweet potato + steamed veg\n\n"
            f"**Pro Tips:**\n"
            f"• Water: **{water}L daily** ({w}kg × 0.033)\n"
            f"• Eat protein within 30 min post-workout\n"
            f"• Meal prep Sunday → saves time + money 📦"
        )

    # ── CALORIES / TDEE / BMR ────────────────────────────────────────────
    if _match(t, ["calori","tdee","bmr","how much should i eat","maintenance","daily calori"]):
        bmr_val = round(calc_tdee(w, h, age, gender, 1) / 1.2)   # reverse TDEE to get BMR
        return (
            f"**Your Calorie Calculations** 🔢\n\n"
            f"**BMR (Base Metabolic Rate): {bmr_val} kcal**\n"
            f"→ Calories burned doing absolutely nothing (lying in bed all day)\n\n"
            f"**TDEE (Total Daily Energy Expenditure): {tdee} kcal**\n"
            f"→ Actual calories burned with your activity level ({act}/5)\n\n"
            f"**Your Targets by Goal:**\n"
            f"• 🔥 Lose weight (−0.5kg/wk): **{tdee-500} kcal**\n"
            f"• ⚖️  Maintain weight:           **{tdee} kcal**\n"
            f"• 💪 Build muscle (+0.25kg/wk): **{tdee+300} kcal**\n\n"
            f"**Formula used:** Mifflin-St Jeor (most accurate for general population)\n"
            f"**Your current goal target: {cals} kcal/day**"
        )

    # ── BMI ──────────────────────────────────────────────────────────────
    if _match(t, ["bmi","body mass index","am i overweight","healthy weight"]):
        ideal_low  = round((h/100)**2 * 18.5, 1)
        ideal_high = round((h/100)**2 * 24.9, 1)
        return (
            f"**Your BMI Analysis** 📊\n\n"
            f"• BMI: **{bmi}** — **{bmi_cat}**\n"
            f"• Height: {h}cm | Weight: {w}kg\n"
            f"• Healthy weight range for your height: **{ideal_low}–{ideal_high}kg**\n\n"
            f"**WHO BMI Scale:**\n"
            f"• < 18.5  → Underweight\n"
            f"• 18.5–24.9 → Normal weight ✅\n"
            f"• 25–29.9 → Overweight\n"
            f"• 30+     → Obese\n\n"
            f"{'✅ You are in the healthy range!' if 18.5 <= bmi < 25 else f'⚠️  Your goal target of {t_w}kg will bring your BMI to {calc_bmi(t_w, h)}'}"
        )

    # ── PROTEIN / SUPPLEMENTS ────────────────────────────────────────────
    if _match(t, ["how much protein","protein per day","protein intake","daily protein","protein goal","protein need","how many protein"]):
        return (
            f"**Protein Requirements for {name}** 🥩\n\n"
            f"**Your Daily Protein Target: {protein}g**\n"
            f"*(Formula: {w}kg × 2.0 = {protein}g)*\n\n"
            f"**Why protein matters:**\n"
            f"• Builds and repairs muscle after training\n"
            f"• Keeps you full longer (most satiating macro)\n"
            f"• Preserves muscle during fat loss\n"
            f"• Has the highest thermic effect (burns ~25% of its own calories digesting it)\n\n"
            f"**Best protein sources:**\n"
            f"• Chicken breast  — 31g per 100g\n"
            f"• Tuna (canned)   — 30g per 100g\n"
            f"• Lean beef       — 26g per 100g\n"
            f"• Eggs            — 6g each\n"
            f"• Greek yoghurt   — 10g per 100g\n"
            f"• Whey protein    — 25g per scoop\n\n"
            f"**Spread it across meals:** ~{round(protein/4)}g per meal (4 meals)\n"
            f"**Post-workout:** eat {round(protein*0.25)}g within 30 minutes 💪"
        )

    if _match(t, ["protein","supplement","creatine","whey","bcaa","pre-workout","vitamin","fish oil","omega"]):
        return (
            f"**Supplement Guide for {name}** 💊\n\n"
            f"**🥇 Tier 1 — Essential (always worth it):**\n"
            f"• **Creatine Monohydrate** 5g/day — increases strength 5–15% guaranteed ✅\n"
            f"• **Whey Protein** — {round(w*0.3)}g post-workout to hit your {protein}g target\n"
            f"• **Vitamin D3** 2000–4000 IU — most people are deficient\n\n"
            f"**🥈 Tier 2 — Highly Beneficial:**\n"
            f"• **Omega-3 Fish Oil** 2–3g EPA+DHA — reduces inflammation\n"
            f"• **Magnesium Glycinate** 300mg before bed — better sleep & recovery\n"
            f"• **Zinc** 15–30mg — testosterone and immune function\n\n"
            f"**🥉 Tier 3 — Optional:**\n"
            f"• **Pre-workout** — useful for energy, but not essential\n"
            f"• **BCAAs** — only useful if training fasted with low protein intake\n"
            f"• **Caffeine** — {round(w*3)}–{round(w*6)}mg, 30 min before training\n\n"
            f"⚠️ *Food first. Supplements only enhance a good diet — they cannot fix a bad one.*"
        )

    # ── FAT LOSS ─────────────────────────────────────────────────────────
    if _match(t, ["lose weight","fat loss","weight loss","cutting","slim","drop weight"]):
        return (
            f"**Fat Loss Blueprint for {name}** 🔥\n"
            f"*{w}kg → {t_w}kg | Est. {weeks_to_goal} weeks at 0.5kg/week*\n\n"
            f"**The Science:**\n"
            f"• 1kg fat = 7,700 kcal deficit\n"
            f"• Safe rate: 0.5–1kg/week\n"
            f"• Your daily target: **{cals} kcal** ({tdee} − 500)\n\n"
            f"**The 5-Step System:**\n"
            f"1. **Eat {cals} kcal/day** — use MyFitnessPal to track\n"
            f"2. **Eat {protein}g protein** — stops muscle loss while cutting\n"
            f"3. **Lift weights 3×/week** — preserves muscle\n"
            f"4. **Walk 8,000 steps/day** — easiest calorie burn\n"
            f"5. **Sleep 7–9 hours** — cortisol destroys fat loss\n\n"
            f"**Common Mistakes:**\n"
            f"• Eating too little (<1200 kcal) — slows metabolism\n"
            f"• Only doing cardio — you will lose muscle, not just fat\n"
            f"• Weighing daily — weight fluctuates 1–3kg from water alone\n\n"
            f"📊 Weigh weekly, same time, same conditions. Trust the process!"
        )

    # ── MUSCLE BUILDING ──────────────────────────────────────────────────
    if _match(t, ["build muscle","gain muscle","bulk","bulking","hypertrophy","mass","get bigger"]):
        return (
            f"**Muscle Building Blueprint for {name}** 💪\n"
            f"*Current: {w}kg | Bulk calories: {cals} kcal/day*\n\n"
            f"**The Science of Hypertrophy:**\n"
            f"• Muscle needs: progressive overload + protein + sleep\n"
            f"• Realistic gain: **0.5–1kg muscle/month** (natural)\n"
            f"• Your calorie surplus: +{cals-tdee} kcal above maintenance\n\n"
            f"**The 3 Non-Negotiables:**\n"
            f"1. **Progressive overload** — add weight or reps every week\n"
            f"2. **{protein}g protein daily** — 2g per kg bodyweight\n"
            f"3. **8 hours sleep** — growth hormone peaks during deep sleep\n\n"
            f"**Optimal Rep Ranges:**\n"
            f"• 3–5 reps   → Maximum strength\n"
            f"• 6–12 reps  → **Hypertrophy zone** ← focus here\n"
            f"• 15–20 reps → Muscular endurance\n\n"
            f"**The Big 6 Movements (master these):**\n"
            f"Squat · Deadlift · Bench Press · OHP · Pull-ups · Rows"
        )

    # ── CARDIO ───────────────────────────────────────────────────────────
    if _match(t, ["cardio","running","hiit","liss","cycling","treadmill","fat burn zone","aerobic","cardio tips"]):
        return (
            f"**Cardio Guide for {name}** 🏃\n"
            f"*Weight: {w}kg | Age: {age} | Goal: {goal}*\n\n"
            f"**HIIT (High Intensity Interval Training):**\n"
            f"• Burns ~{round(w*0.1*8)} kcal in 20 min\n"
            f"• Format: 40s all-out / 20s rest × 15 rounds\n"
            f"• Heart rate target: **{hr['peak'][0]}–{hr['peak'][1]} BPM**\n"
            f"• Do 2–3× per week (not on leg day!)\n\n"
            f"**LISS (Low Intensity Steady State):**\n"
            f"• Burns ~{round(w*0.05*45)} kcal in 45 min\n"
            f"• Heart rate zone: **{hr['fat_burn'][0]}–{hr['fat_burn'][1]} BPM** (fat-burn zone)\n"
            f"• Walking, cycling, swimming — great on rest days\n\n"
            f"**Your Max Heart Rate: {hr['max_hr']} BPM** (220 − {age})\n\n"
            f"**Recommended for {goal}:**\n"
            + ("3× HIIT + 3× 45-min walks per week 🔥" if goal == "lose"
               else "2× light cardio/week only — protect your muscle gains!" if goal == "build"
               else "2–3× mixed cardio per week — choose what you enjoy!")
        )

    # ── SLEEP & RECOVERY ─────────────────────────────────────────────────
    if _match(t, ["sleep","recovery","rest day","sore","doms","overtraining","fatigue","tired"]):
        return (
            f"**Recovery & Sleep Guide for {name}** 💤\n\n"
            f"*Muscle grows OUTSIDE the gym — during sleep and rest*\n\n"
            f"**Sleep Optimisation:**\n"
            f"• Target: **8 hours** per night (minimum 7)\n"
            f"• Consistent sleep/wake times — even weekends\n"
            f"• Room temp 18–20°C for optimal sleep quality\n"
            f"• No screens 30 min before bed (blue light disrupts melatonin)\n"
            f"• **Magnesium Glycinate 300mg** before bed → deeper sleep\n\n"
            f"**Muscle Soreness (DOMS):**\n"
            f"• Normal for 24–72h after new exercises\n"
            f"• Light movement speeds recovery — short walk or stretch\n"
            f"• Ice for acute injury, heat for chronic soreness\n\n"
            f"**Active Recovery Days:**\n"
            f"• 20–30 min light walk | Foam roll 10 min | Yoga 15 min\n\n"
            f"**Signs of Overtraining:**\n"
            f"• Persistent fatigue, dropping performance, poor sleep\n"
            f"→ Take a **deload week** — cut volume by 50%, keep intensity"
        )

    # ── INTERMITTENT FASTING ─────────────────────────────────────────────
    if _match(t, ["intermittent fasting","16:8","18:6","fasting","eating window","skip breakfast"]):
        return (
            f"**Intermittent Fasting for {name}** ⏰\n\n"
            f"**What is IF?** Restricting eating to a set time window each day.\n\n"
            f"**Most Popular Protocols:**\n"
            f"• **16:8** — Fast 16h, eat in 8h window (e.g. 12pm–8pm)\n"
            f"• **18:6** — More aggressive, better for fat loss\n"
            f"• **5:2**  — Normal 5 days, ~500 kcal on 2 days\n\n"
            f"**Your IF Numbers:**\n"
            f"• Eating window calories: **{cals} kcal**\n"
            f"• Protein per meal: ~{round(protein/3)}g (spread over 2–3 meals)\n\n"
            f"**During the Fast (allowed):**\n"
            f"• Water ✅ | Black coffee ✅ | Plain tea ✅ | Zero-cal drinks ✅\n\n"
            f"**Benefits:** Simplifies tracking · Improves insulin sensitivity · Reduces overall intake\n\n"
            f"⚠️ *Not recommended if you have a history of disordered eating.*"
        )

    # ── HYDRATION ────────────────────────────────────────────────────────
    if _match(t, ["water","hydration","drink","hydrate","how much water"]):
        return (
            f"**Hydration Guide for {name}** 💧\n\n"
            f"**Daily Water Target: {water}L**\n"
            f"*(Formula: {w}kg × 0.033 = {water}L)*\n\n"
            f"**How to hit {water}L daily:**\n"
            f"• 500ml immediately on waking\n"
            f"• 500ml before each of 3 meals\n"
            f"• 750ml during your workout\n"
            f"• Remaining sipped throughout the day\n\n"
            f"**Performance impact:**\n"
            f"• 2% dehydration → 10–20% drop in performance\n"
            f"• Thirst = already slightly dehydrated\n"
            f"• Add 500ml per 30 min of intense exercise in heat\n\n"
            f"**Check your hydration:** Urine should be pale yellow 🟡\n"
            f"Dark yellow = drink more | Colourless = drink less"
        )

    # ── FORM & TECHNIQUE ─────────────────────────────────────────────────
    if _match(t, ["form","technique","how to squat","how to deadlift","how to bench",
                  "proper form","posture","correct form"]):
        return (
            f"**Exercise Form Guide** 🎯\n\n"
            f"*Form = more gains + zero injuries. Always prioritise technique over weight!*\n\n"
            f"**Squat:**\n"
            f"• Feet shoulder-width, toes 15–30° out\n"
            f"• Chest up, core braced, neutral spine\n"
            f"• Knees track over toes, break parallel\n"
            f"• Drive through heels to stand\n\n"
            f"**Deadlift:**\n"
            f"• Bar over mid-foot, hip-width stance\n"
            f"• Hinge at hips — flat back (no rounding!)\n"
            f"• Bar stays in contact with legs throughout\n"
            f"• Lock out hips at top — don't hyperextend\n\n"
            f"**Bench Press:**\n"
            f"• 5-point contact (head, traps, butt, both feet flat)\n"
            f"• Retract scapula before unracking\n"
            f"• Touch lower chest, press up and slightly back\n\n"
            f"**Pull-up:**\n"
            f"• Full hang, shoulder-width grip\n"
            f"• Pull elbows to hips — not chin over bar\n"
            f"• Lower slowly (4-second eccentric = more gains)\n\n"
            f"💡 *Use APEX AI camera mode for real-time AI form feedback!*"
        )

    # ── MOTIVATION ───────────────────────────────────────────────────────
    if _match(t, ["motivat","lazy","give up","giving up","plateau","not working","struggling","discouraged","stuck","no progress","want to quit","feel like quitting","losing hope","nothing is working","hate working out"]):
        return (
            f"Hey {name} — I hear you, and I want you to know: **every champion felt exactly this way.** 🔥\n\n"
            f"**Breaking Your Plateau:**\n"
            f"• Change your rep scheme this week (try 5×5 if you've been doing 3×10)\n"
            f"• Add 1 new exercise you have never done before\n"
            f"• Deload for 1 week — cut volume 50%, this often IS the breakthrough\n\n"
            f"**The truth about progress:**\n"
            f"• It is never linear — flat weeks happen to EVERYONE\n"
            f"• Strength is still increasing even when weight does not move\n"
            f"• Photos and measurements beat the scale every time\n\n"
            f"**Your stats right now:**\n"
            f"• You burn **{tdee} kcal** just existing — every workout adds on top\n"
            f"• You are only **{round(abs(w - t_w), 1)}kg** from your goal weight\n"
            f"• Estimated time to goal at 0.5kg/week: **{weeks_to_goal} weeks**\n\n"
            f"**Action for right now:** Just 15 minutes. Start. The rest follows. 💪\n\n"
            f"I believe in you, {name}. You've got this! 🚀"
        )

    # ── DEFAULT ─────────────────────────────────────────────────────────
    return (
        f"Hey {name}! 💪 Here are your quick stats:\n\n"
        f"• Calories: **{cals} kcal/day** | Protein: **{protein}g/day**\n"
        f"• BMI: **{bmi}** ({bmi_cat}) | Goal: **{goal}**\n\n"
        f"Ask me anything about fitness! For example:\n"
        f"🏋️ *\"Give me a workout plan\"*\n"
        f"🥗 *\"Create my meal plan\"*\n"
        f"💊 *\"What supplements do I need?\"*\n"
        f"🔥 *\"How do I lose fat fast?\"*\n"
        f"📊 *\"Calculate my BMI and TDEE\"*\n"
        f"💤 *\"How do I recover faster?\"*"
    )


def _match(text: str, keywords: list) -> bool:
    """Return True if any keyword is found in the text.
    Handles both single-word (uses word boundary) and
    multi-word keywords (uses plain substring search).
    """
    import re
    for kw in keywords:
        if " " in kw:
            # Multi-word: plain substring match
            if kw in text:
                return True
        else:
            # Single word: word boundary match (avoids partial matches)
            if re.search(r'\b' + re.escape(kw), text):
                return True
    return False
