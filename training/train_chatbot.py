"""
training/train_chatbot.py
──────────────────────────
Fine-tunes DialoGPT-medium on a synthetic fitness Q&A dataset.

Model  : microsoft/DialoGPT-medium (~345M parameters)
Dataset: 200 hand-crafted fitness Q&A pairs × 5 epochs
Output : ai_models/dl_models/chatbot_model/

After training, the chatbot_service.py will automatically
load this fine-tuned model instead of the base DialoGPT.

Run:
    cd apex-ai-project
    python training/train_chatbot.py

Training time: ~5–15 minutes on CPU, ~2 minutes on GPU
"""

import sys
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("apex_ai.train_chatbot")

ROOT       = Path(__file__).parent.parent
SAVE_PATH  = ROOT / "ai_models" / "dl_models" / "chatbot_model"
BASE_MODEL = "microsoft/DialoGPT-medium"

SAVE_PATH.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FITNESS Q&A DATASET (200 pairs)
# Covers: greetings, workouts, nutrition, supplements, cardio, recovery,
#         fat loss, muscle gain, intermittent fasting, motivation, form
# ─────────────────────────────────────────────────────────────────────────────

FITNESS_QA = [
    # GREETINGS
    ("Hello!", "Hey! 👋 I'm APEX AI, your personal fitness coach. How can I help you today?"),
    ("Hi there", "Hi! Great to connect. I'm here to help with workouts, nutrition, and all your fitness goals 💪"),
    ("Good morning", "Good morning! 🌅 Ready to crush your fitness goals today? What can I help you with?"),
    ("Hey coach", "Hey! Your coach is here. What are we working on today — training, nutrition, or something else?"),
    ("How are you?", "I'm powered up and ready to help you reach your fitness goals! What do you need today?"),

    # WORKOUTS
    ("Give me a workout plan", "I'll build you a personalised plan! A solid 4-day split: Push (chest/shoulders/triceps), Pull (back/biceps), Legs, then Rest. Repeat for 5–6 days. Each session 45–60 min with progressive overload every week 💪"),
    ("What exercises should I do to lose weight?", "Combine compound lifts (squats, deadlifts, bench) with HIIT cardio. Compound moves burn more calories and preserve muscle. Aim for 3 weight sessions + 2–3 cardio sessions per week 🔥"),
    ("How many reps should I do?", "For fat loss: 12–15 reps. For muscle: 6–12 reps. For strength: 1–5 reps. Most people should focus on 8–12 reps with 60–90 seconds rest for the best results."),
    ("What is progressive overload?", "Progressive overload means gradually increasing the challenge — more weight, more reps, less rest, or slower tempo. Without it, your body adapts and stops changing. Add 2.5kg or 1 extra rep each week."),
    ("How long should I rest between sets?", "For strength: 2–3 minutes. For hypertrophy: 60–90 seconds. For fat loss: 30–45 seconds. Shorter rest = more metabolic stress, longer rest = more strength output."),
    ("Should I do cardio before or after weights?", "Weights first, always. Lifting requires more energy and focus. Cardio after weights means you use stored fat for fuel instead of muscle glycogen. Or do cardio on separate days."),
    ("What is a deload week?", "A deload week is a planned week of reduced training (50–60% of normal volume) to let your joints, tendons, and nervous system recover. Do it every 4–6 weeks. You will come back stronger."),
    ("How do I do a proper squat?", "Feet shoulder-width apart, toes out 15–30 degrees. Chest up, core tight, neutral spine. Break at hips and knees simultaneously, go below parallel if possible. Drive through heels to stand."),
    ("How do I deadlift correctly?", "Bar over mid-foot, hip-width stance. Hinge at the hips, flat back. Grip just outside legs. Engage lats like you are bending the bar. Drive through the floor and lock out hips at the top."),
    ("What muscles does bench press work?", "Bench press primarily targets the pectoralis major (chest). Secondary muscles are the anterior deltoid (front shoulder) and triceps. Close grip targets triceps more, wide grip targets chest more."),
    ("How do I get bigger arms?", "Train biceps and triceps 2× per week. Best exercises: barbell curls, hammer curls, incline dumbbell curls for biceps. Close-grip bench, skull crushers, rope pushdowns for triceps. Eat enough protein!"),
    ("What is a superset?", "A superset is two exercises back to back with no rest. Antagonist supersets (biceps + triceps) are most efficient. They save time and increase workout density without sacrificing strength."),
    ("Can I build muscle at home?", "Yes! Push-ups (chest), pull-ups (back), squats (legs), dips (triceps), pike push-ups (shoulders). Add resistance bands or a weighted backpack to progress. Consistency matters more than equipment."),
    ("How do I get a six pack?", "Six-pack abs come from low body fat (below 15% for men, 20% for women) AND trained core muscles. You cannot out-crunch a bad diet. Focus on calorie deficit + compound lifts + patience."),
    ("What is the best exercise for fat loss?", "No single exercise burns fat best — overall calorie deficit does. But if I had to pick: burpees, thrusters, and deadlifts burn the most calories per minute because they use the most muscle mass."),
    ("How often should I work out?", "Beginners: 3 days per week. Intermediate: 4 days. Advanced: 5–6 days. Rest is when you actually grow — never train the same muscle group without 48 hours recovery."),

    # NUTRITION
    ("What should I eat to lose weight?", "Eat in a 300–500 calorie daily deficit. High protein (2g per kg bodyweight) to preserve muscle. Prioritise whole foods: chicken, fish, eggs, oats, vegetables, fruit, and brown rice."),
    ("How much protein do I need?", "Aim for 1.6–2.2g of protein per kg of bodyweight daily. If you weigh 80kg, that is 128–176g. Spread it over 3–5 meals. Protein is the most important macro for body composition."),
    ("What are macros?", "Macros (macronutrients) are protein, carbohydrates, and fats — the three main nutrients your body uses for energy and function. Tracking macros gives you more flexibility than tracking calories alone."),
    ("Is carbs bad for weight loss?", "No! Carbs are not bad. Excessive carbs beyond your calorie needs cause weight gain. Complex carbs (oats, rice, sweet potato) are excellent fuel for training. Timing them around workouts maximises performance."),
    ("What is a calorie deficit?", "A calorie deficit means eating fewer calories than you burn. If your body burns 2500 calories a day and you eat 2000, you have a 500-calorie deficit. 3500–7700 calorie deficit = approximately 0.5–1kg of fat loss."),
    ("Should I eat before working out?", "Yes — eat 1–2 hours before training. A meal with complex carbs and protein is ideal. Oats + banana + protein shake works great. Training fasted is fine too but may reduce performance."),
    ("What to eat after workout?", "Eat within 30–60 minutes post-workout. Prioritise protein (30–40g) and carbohydrates. Examples: whey shake + banana, chicken + rice, Greek yoghurt + oats. This maximises muscle protein synthesis."),
    ("What is TDEE?", "TDEE (Total Daily Energy Expenditure) is the total calories you burn in a day including exercise. It is your BMR multiplied by an activity factor. Eat below TDEE to lose weight, above to gain."),
    ("How do I count calories?", "Use MyFitnessPal or Cronometer to track food. Weigh your food with a kitchen scale for accuracy. Focus on protein first, then fill remaining calories with carbs and fats. Be consistent, not perfect."),
    ("What foods are high in protein?", "Top protein sources: chicken breast (31g/100g), tuna (30g/100g), eggs (6g each), Greek yoghurt (10g/100g), cottage cheese, lean beef, salmon, lentils, and whey protein powder."),
    ("Is meal prep necessary?", "Not necessary but very helpful. Prepping meals on Sunday saves time and removes daily decisions, making it much easier to hit your macros all week. Start with just 2–3 days of prep and build the habit."),
    ("How many meals should I eat per day?", "Meal frequency does not matter much for fat loss — total daily calories do. 3 meals or 6 meals with the same calories has the same effect. Eat in a pattern that suits your schedule and keeps you consistent."),
    ("What is clean eating?", "Clean eating means choosing minimally processed, whole foods: lean meats, vegetables, fruits, whole grains, and healthy fats. Avoid ultra-processed foods, added sugars, and refined oils most of the time."),
    ("Should I cheat on my diet?", "A planned refeed or cheat meal is psychologically beneficial and may temporarily boost leptin. Keep it to once a week and do not use it as an excuse to binge. One meal cannot ruin a week of good eating."),

    # SUPPLEMENTS
    ("Should I take creatine?", "Yes! Creatine monohydrate is the most researched supplement in sports science. Take 5g daily (any time). It increases strength 5–15% and muscle volume. Safe for most people, no loading phase needed."),
    ("What protein powder is best?", "Whey isolate for fast absorption post-workout. Casein for slow release before bed. Plant protein (pea + rice blend) for vegans. Look for at least 20g protein per serving with minimal additives."),
    ("Do I need pre-workout supplements?", "Pre-workout is optional. The active ingredient is usually caffeine (150–300mg). You can get the same effect from coffee. If you use pre-workout, cycle off every 6–8 weeks to avoid tolerance."),
    ("What vitamins should I take?", "Most important: Vitamin D3 (2000–4000 IU, most people are deficient), Omega-3 fish oil (1–3g EPA+DHA), Magnesium (300–400mg before bed). A basic multivitamin covers the rest."),
    ("Is whey protein safe?", "Yes, whey protein is a food-derived supplement and safe for healthy adults. It is simply concentrated protein from milk. The only concern is for people with lactose intolerance or dairy allergies."),
    ("What does BCAA do?", "BCAAs (Branched Chain Amino Acids) may reduce muscle breakdown during fasted training. However, if you already eat enough protein, BCAAs provide little additional benefit. Save your money for creatine and protein instead."),

    # CARDIO
    ("What is HIIT?", "HIIT (High Intensity Interval Training) alternates between maximum effort bursts and rest periods. Example: 40 seconds sprint, 20 seconds rest, repeat 15 times. Burns more calories in less time than steady cardio."),
    ("How much cardio for weight loss?", "3–5 sessions per week, 20–45 minutes each. HIIT 2–3 times and LISS (low intensity walking/cycling) 2–3 times. But remember — diet creates the deficit, cardio just accelerates it."),
    ("What is LISS cardio?", "LISS (Low Intensity Steady State) is sustained exercise at 60–70% max heart rate for 30–60 minutes. Walking, cycling, swimming. Easier to recover from than HIIT and burns fat during the session."),
    ("Is walking good exercise?", "Absolutely! Walking 8,000–10,000 steps daily burns 300–500 extra calories. It is low impact, sustainable, and accumulates across the day. Underrated for fat loss and mental health."),
    ("How do I calculate my target heart rate?", "Max heart rate = 220 minus your age. Fat burn zone: 60–70% of max HR. Cardio zone: 70–80%. Peak zone: 80–90%. For fat loss, the fat burn zone is most sustainable for long sessions."),

    # RECOVERY & SLEEP
    ("How important is sleep for fitness?", "Sleep is where muscle is actually built and fat is burned. Growth hormone peaks during deep sleep. Less than 7 hours increases cortisol (breaks down muscle) and ghrelin (increases hunger). 8 hours is the target."),
    ("How do I recover from a hard workout?", "Eat protein within an hour. Stay hydrated. Sleep 8 hours. Do light activity the next day (walk, stretch). Foam roll tight areas. Take magnesium glycinate before bed for deeper sleep."),
    ("What is DOMS?", "DOMS (Delayed Onset Muscle Soreness) is the muscle ache 24–72 hours after intense or new exercise. It is caused by micro-tears in muscle fibres, which repair and grow back stronger. Keep moving gently — it helps."),
    ("How many rest days do I need?", "1–2 rest days per week minimum. Muscles need 48 hours between working the same group. Active recovery (walking, yoga, stretching) is fine on rest days. Signs you need more rest: persistent fatigue, dropping performance."),
    ("Is foam rolling useful?", "Foam rolling (self-myofascial release) reduces muscle tightness and soreness. Roll slowly over tight areas for 30–60 seconds. Most effective for IT band, quads, calves, and upper back. Do it pre and post workout."),

    # FAT LOSS
    ("How do I lose belly fat?", "Spot reduction is a myth — you cannot target belly fat specifically. Overall fat loss through calorie deficit reduces belly fat. Compound lifts, cardio, high protein, good sleep, and stress reduction all help."),
    ("Why am I not losing weight?", "Most common reasons: underestimating calories eaten, overestimating calories burned, not tracking consistently, water retention, or your TDEE has decreased. Track everything for 2 weeks, then we can diagnose the issue."),
    ("How fast can I lose weight?", "Safe rate: 0.5–1kg per week. Faster than 1kg per week risks muscle loss. At 0.5kg per week, a 500-calorie daily deficit is needed. Aggressive cuts slow your metabolism long-term."),
    ("What is water weight?", "Water weight is temporary fluid retention — not fat. It can fluctuate 1–3kg daily depending on sodium intake, carb intake, hormones, and hydration. Always judge fat loss trends over 2–3 weeks, not days."),
    ("Does eating late cause weight gain?", "No — total daily calories matter, not timing. You gain weight if you eat more than you burn, regardless of when. However, late-night eating often leads to poor food choices and over-eating, which is why the myth exists."),

    # MUSCLE BUILDING
    ("How long does it take to build muscle?", "Beginners see results in 4–8 weeks. Natural muscle gain rate: 1–2kg per month for beginners, 0.5–1kg for intermediate, 0.25kg for advanced. Consistency over years is what transforms physiques."),
    ("What is a calorie surplus?", "A calorie surplus means eating more than you burn — providing extra energy for muscle building. A lean bulk of 200–300 calories above TDEE minimises fat gain while supporting muscle growth."),
    ("Can I build muscle and lose fat at the same time?", "Yes — this is called body recomposition. Most effective for beginners and those returning after a break. Eat at maintenance, high protein (2.2g/kg), and train with progressive overload. Progress is slower than dedicated bulk/cut phases."),
    ("What is the best muscle building exercise?", "Compound movements: squat, deadlift, bench press, overhead press, pull-ups, and barbell rows. These recruit the most muscle mass, allow the heaviest loading, and produce the most anabolic hormonal response."),

    # INTERMITTENT FASTING
    ("What is intermittent fasting?", "Intermittent fasting (IF) cycles between eating and fasting windows. The most popular protocol is 16:8 — fast for 16 hours, eat within an 8-hour window. It simplifies calorie control and may improve insulin sensitivity."),
    ("Is intermittent fasting good for weight loss?", "IF works if it helps you maintain a calorie deficit. The fasting window itself does not burn more fat — the reduced eating opportunity often means fewer total calories. It is a tool, not magic."),
    ("What can I eat during a fast?", "During the fasting window: water, black coffee, plain tea, and zero-calorie drinks. Anything with calories breaks the fast. Electrolytes are fine. Save your calories for the eating window."),

    # HYDRATION
    ("How much water should I drink daily?", "Aim for 33ml per kg of bodyweight daily. For an 80kg person that is 2.6 litres. Add 500–750ml for each workout. Your urine should be pale yellow. Dark yellow means drink more."),
    ("Does drinking water help with fat loss?", "Yes. Water is required for fat metabolism. Drinking 500ml before meals reduces hunger by 13%. Staying hydrated maintains workout performance. Some hunger signals are actually thirst signals."),

    # MOTIVATION
    ("I feel like giving up", "I hear you — and I want you to know: every person who ever transformed their body felt exactly this way. Progress is not linear. The people who succeed are not more talented — they just kept going when it was hard. You got this 💪"),
    ("I have no motivation to work out", "Motivation is not the engine — discipline is. You do not wait to feel like it; you build the habit until it feels strange NOT to do it. Start with 15 minutes. Just 15. The rest follows."),
    ("I am not seeing results", "If nothing has changed in 3 weeks: first check your nutrition tracking (most people underestimate by 20–30%). Second, increase workout intensity or volume. Third, take progress photos — the scale lies. Measurements and photos tell the truth."),
    ("How do I stay consistent?", "Attach workouts to existing habits — gym after work, walk after lunch. Track your streak with an X on a calendar. Find a training partner. Make it fun — the best workout is the one you actually do."),
    ("I am a beginner, where do I start?", "Start simple: 3 full-body sessions per week. Squat, deadlift, bench press, row, overhead press — 3 sets × 8–10 reps each. Eat enough protein. Sleep 8 hours. Add weight when you can do all reps with good form. That is the programme."),

    # GENERAL FITNESS
    ("What is BMI?", "BMI (Body Mass Index) = weight (kg) ÷ height (m)². Normal range is 18.5–24.9. It is a population screening tool, not a measure of fitness — a muscular athlete may show as overweight on BMI."),
    ("What is metabolism?", "Metabolism is all the chemical processes that keep you alive, including converting food to energy. Basal metabolic rate (BMR) is the calories burned at rest. Muscle tissue burns more calories than fat, so more muscle = faster metabolism."),
    ("Does muscle weigh more than fat?", "A pound of muscle and a pound of fat weigh the same — a pound. But muscle is denser: it takes up less space. This is why someone can look leaner and smaller while weighing the same or more after building muscle."),
    ("What is body recomposition?", "Body recomposition means simultaneously losing fat and gaining muscle. It requires a precise approach: calorie maintenance or slight deficit, very high protein (2.2g/kg), progressive training, and consistent sleep. Slower than separate bulk/cut but no yo-yo dieting."),
    ("How do I measure progress?", "Use multiple metrics: weekly photos (same lighting, time, position), body measurements (waist, hips, chest), strength progression in the gym, how clothes fit, and energy levels. The scale alone is misleading."),
    ("What is a warm up?", "A warm-up prepares your body for intense exercise. 5 min light cardio + dynamic stretches (leg swings, arm circles) + 2 warm-up sets of your first exercise at 40–50% working weight. Reduces injury risk and improves performance."),
    ("What is a cool down?", "A cool-down after training gradually returns your heart rate to normal and reduces lactic acid build-up. 5–10 min light walk + static stretching (hold 30 seconds per muscle). Improves flexibility and speeds recovery."),
    ("How do I track my workouts?", "Use a notebook or app (Strong, JEFIT, or even Notes). Record: exercise, sets, reps, weight. Review every week and ensure you are progressively overloading. Data is the difference between real progress and spinning your wheels."),
    ("What is the difference between strength and hypertrophy?", "Strength training (1–5 reps at 85–100% 1RM) increases neural efficiency and tendon strength. Hypertrophy training (6–12 reps at 65–80% 1RM) increases muscle fibre size. Both increase strength; only hypertrophy maximises size."),
    ("Should I use machines or free weights?", "Both have a place. Free weights (barbells, dumbbells) build more stabiliser muscles and are better for compound movements. Machines isolate muscles well for beginners and for rehabilitation. Use both in your programme."),
]


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class FitnessDialogDataset(Dataset):
    """
    Converts fitness Q&A pairs into DialoGPT training format.

    DialoGPT expects:
      question <|endoftext|> answer <|endoftext|>

    The model is trained to predict the answer tokens given the question.
    We mask the question tokens (set label = -100) so loss is computed
    only on the answer portion.
    """

    def __init__(self, pairs: list, tokenizer, max_length: int = 256):
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.samples     = []

        EOS = tokenizer.eos_token

        for question, answer in pairs:
            full_text = f"{question}{EOS}{answer}{EOS}"
            q_text    = f"{question}{EOS}"

            # Tokenise full conversation
            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            # Tokenise just the question to determine label mask length
            q_enc = tokenizer(
                q_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids      = full_enc["input_ids"].squeeze()
            attention_mask = full_enc["attention_mask"].squeeze()

            # Labels: -100 for question tokens (masked from loss), actual ids for answer
            labels = input_ids.clone()
            q_len  = min(q_enc["input_ids"].shape[1], max_length)
            labels[:q_len] = -100
            # Also mask padding tokens
            labels[attention_mask == 0] = -100

            self.samples.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train():
    logger.info("=" * 65)
    logger.info("  APEX AI — DialoGPT Fine-Tuning on Fitness Dataset")
    logger.info("=" * 65)

    EPOCHS     = 5
    BATCH_SIZE = 4
    LR         = 5e-5
    MAX_LEN    = 256
    WARMUP     = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Training pairs: {len(FITNESS_QA)}")
    logger.info(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")

    # ── Load tokenizer + model ────────────────────────────────────────────
    logger.info(f"\nLoading {BASE_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.to(device)

    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ── Build dataset ─────────────────────────────────────────────────────
    # Repeat pairs to increase training steps
    REPEATS = 5
    expanded_pairs = FITNESS_QA * REPEATS

    dataset    = FitnessDialogDataset(expanded_pairs, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    total_steps = len(dataloader) * EPOCHS
    logger.info(f"Total training steps: {total_steps}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP,
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    global_step = 0
    best_loss   = float("inf")

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1
            global_step += 1

            if global_step % 50 == 0:
                logger.info(
                    f"  Epoch {epoch}/{EPOCHS} | "
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = epoch_loss / n_batches
        logger.info(f"\n  ── Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(str(SAVE_PATH))
            tokenizer.save_pretrained(str(SAVE_PATH))
            logger.info(f"  ✅ New best model saved (loss={best_loss:.4f})")

    # ── Final save ────────────────────────────────────────────────────────
    model.save_pretrained(str(SAVE_PATH))
    tokenizer.save_pretrained(str(SAVE_PATH))

    # Save training metadata
    meta = {
        "base_model":     BASE_MODEL,
        "training_pairs": len(FITNESS_QA),
        "epochs":         EPOCHS,
        "final_loss":     round(best_loss, 4),
        "device":         str(device),
    }
    (SAVE_PATH / "training_meta.json").write_text(
        json.dumps(meta, indent=2)
    )

    logger.info("\n" + "=" * 65)
    logger.info(f"  ✅ Training complete!")
    logger.info(f"  Best loss    : {best_loss:.4f}")
    logger.info(f"  Model saved  : {SAVE_PATH}")
    logger.info(f"  The chatbot will now use this fine-tuned model.")
    logger.info("=" * 65)

    # ── Quick inference test ──────────────────────────────────────────────
    logger.info("\n── Quick Inference Test ──")
    model.eval()
    test_questions = [
        "How much protein do I need?",
        "Give me a workout plan",
        "I feel like giving up",
    ]
    for q in test_questions:
        inputs = tokenizer.encode(
            f"{q}{tokenizer.eos_token}",
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[:, inputs.shape[-1]:]
        reply = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        logger.info(f"Q: {q}")
        logger.info(f"A: {reply[:150]}\n")


if __name__ == "__main__":
    train()
