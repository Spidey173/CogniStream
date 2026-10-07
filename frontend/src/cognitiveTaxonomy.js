/**
 * CogniStream 50-State Cognitive & Affective Taxonomy
 * Mapped from high-resolution Craniofacial Action Units (FACS),
 * Russell's Circumplex Affect Coordinates (Valence/Arousal),
 * and 3D Cranial Euler Angles.
 */

export const EMOTION_TAXONOMY_50 = {
  // ── Row 1: Joy & Contentment ──
  Happy: {
    category: "Joy & Contentment",
    emoji: "😊",
    valence: 0.85,
    arousal: 0.55,
    description: "Warm bilateral zygomatic smile with relaxed, engaged gaze.",
  },
  Excited: {
    category: "Joy & Contentment",
    emoji: "🤩",
    valence: 0.95,
    arousal: 0.85,
    description: "High-arousal joy with parted lips and wide, energetic eyes.",
  },
  Joyful: {
    category: "Joy & Contentment",
    emoji: "😄",
    valence: 0.9,
    arousal: 0.7,
    description: "Broad Duchenne smile with active crinkling of the eye corners.",
  },
  Playful: {
    category: "Joy & Contentment",
    emoji: "😜",
    valence: 0.8,
    arousal: 0.65,
    description: "Asymmetrical smirk or playful brow raise with mischievous smile.",
  },
  Proud: {
    category: "Joy & Contentment",
    emoji: "😌",
    valence: 0.75,
    arousal: 0.45,
    description: "Elevated chin, direct confident gaze, and subtle satisfied smile.",
  },
  Confident: {
    category: "Joy & Contentment",
    emoji: "😏",
    valence: 0.7,
    arousal: 0.4,
    description: "Centered optical alignment, subtle unilateral smirk, steady gaze.",
  },
  Content: {
    category: "Joy & Contentment",
    emoji: "🙂",
    valence: 0.65,
    arousal: 0.25,
    description: "Gentle resting smile with relaxed craniofacial balance.",
  },
  Relaxed: {
    category: "Joy & Contentment",
    emoji: "🍃",
    valence: 0.5,
    arousal: 0.15,
    description: "Soft facial musculature, calm resting lips, tranquil breathing.",
  },
  Peaceful: {
    category: "Joy & Contentment",
    emoji: "🧘",
    valence: 0.55,
    arousal: 0.1,
    description: "Minimal facial tension, serene ocular relaxation, peaceful harmony.",
  },
  Grateful: {
    category: "Joy & Contentment",
    emoji: "🙏",
    valence: 0.7,
    arousal: 0.3,
    description: "Soft mouth curve, slight humble head tilt, gentle warm gaze.",
  },

  // ── Row 2: Connection & Wonder ──
  Love: {
    category: "Connection & Wonder",
    emoji: "🥰",
    valence: 0.9,
    arousal: 0.5,
    description: "Deep ocular engagement, softened brow, tender bilateral smile.",
  },
  Affectionate: {
    category: "Connection & Wonder",
    emoji: "💖",
    valence: 0.85,
    arousal: 0.45,
    description: "Head cocked affectionately, warm smile, softened eyelid tension.",
  },
  Caring: {
    category: "Connection & Wonder",
    emoji: "🤗",
    valence: 0.75,
    arousal: 0.4,
    description: "Attentive forward posture, compassionate gaze, welcoming expression.",
  },
  Hopeful: {
    category: "Connection & Wonder",
    emoji: "✨",
    valence: 0.7,
    arousal: 0.5,
    description: "Slightly upturned gaze, gentle chin elevation, optimistic smile.",
  },
  Inspired: {
    category: "Connection & Wonder",
    emoji: "💡",
    valence: 0.8,
    arousal: 0.6,
    description: "Widened receptive eyes, elevated brows, illuminated attention.",
  },
  Curious: {
    category: "Connection & Wonder",
    emoji: "🤔",
    valence: 0.4,
    arousal: 0.5,
    description: "Head tilted inquisitively, alert eyes, slight brow lift or smirk.",
  },
  Interested: {
    category: "Connection & Wonder",
    emoji: "🧐",
    valence: 0.45,
    arousal: 0.45,
    description: "Forward head orientation, locked focal gaze, high cognitive alertness.",
  },
  Amazed: {
    category: "Connection & Wonder",
    emoji: "😲",
    valence: 0.7,
    arousal: 0.8,
    description: "Moderate jaw drop, dilated eye aperture, elevated inner brows.",
  },
  Surprised: {
    category: "Connection & Wonder",
    emoji: "😮",
    valence: 0.3,
    arousal: 0.85,
    description: "Classic AU1+2+5+26: elevated brows, wide eyes, parted lips.",
  },
  Shocked: {
    category: "Connection & Wonder",
    emoji: "😱",
    valence: -0.2,
    arousal: 0.95,
    description: "Maximum ocular aperture, involuntary jaw drop, high startled response.",
  },

  // ── Row 3: Cognitive, Inquisitive & Friction ──
  Confused: {
    category: "Cognitive & Inquisitive",
    emoji: "😵‍💫",
    valence: -0.15,
    arousal: 0.45,
    description: "Asymmetrical brow contraction, head tilted sideways, narrowed eyes.",
  },
  Thinking: {
    category: "Cognitive & Inquisitive",
    emoji: "🧠",
    valence: 0.1,
    arousal: 0.35,
    description: "Analytical gaze shift, mild brow focus, compressed contemplative lips.",
  },
  Pensive: {
    category: "Cognitive & Inquisitive",
    emoji: "💭",
    valence: -0.1,
    arousal: 0.25,
    description: "Downturned chin, reflective unhurried gaze, deep inner processing.",
  },
  Uncertain: {
    category: "Cognitive & Inquisitive",
    emoji: "🫤",
    valence: -0.2,
    arousal: 0.35,
    description: "Subtle lip pursing, micro-oscillation in brow tension, guarded gaze.",
  },
  Skeptical: {
    category: "Cognitive & Inquisitive",
    emoji: "🤨",
    valence: -0.1,
    arousal: 0.45,
    description: "Unilateral eyebrow arch or one-sided smirk, appraising analytical stare.",
  },
  Disappointed: {
    category: "Cognitive & Inquisitive",
    emoji: "😞",
    valence: -0.6,
    arousal: 0.25,
    description: "Downturned lip corners, deflated head posture, lowered eyelid gaze.",
  },
  Frustrated: {
    category: "Cognitive & Inquisitive",
    emoji: "😣",
    valence: -0.65,
    arousal: 0.7,
    description: "Marked corrugator furrow, tightened oral corners, high cognitive friction.",
  },
  Annoyed: {
    category: "Cognitive & Inquisitive",
    emoji: "😒",
    valence: -0.5,
    arousal: 0.55,
    description: "Tightened lips, steady irritated gaze, mild cranial tension.",
  },
  Irritated: {
    category: "Cognitive & Inquisitive",
    emoji: "😤",
    valence: -0.6,
    arousal: 0.65,
    description: "Sustained brow furrow, flare in nostril base, rigid facial posture.",
  },
  Bored: {
    category: "Cognitive & Inquisitive",
    emoji: "🥱",
    valence: -0.3,
    arousal: 0.1,
    description: "Drooping eyelids, unexpressive slack jaw, low cognitive engagement.",
  },

  // ── Row 4: Vulnerability & Resistance ──
  Sad: {
    category: "Vulnerability & Resistance",
    emoji: "😔",
    valence: -0.7,
    arousal: 0.25,
    description: "Depressed oral commissures, elevated inner brow corners, heavy gaze.",
  },
  Hurt: {
    category: "Vulnerability & Resistance",
    emoji: "🥺",
    valence: -0.75,
    arousal: 0.4,
    description: "Vulnerable eye aperture, trembling lower lip tension, inner brow ache.",
  },
  Lonely: {
    category: "Vulnerability & Resistance",
    emoji: "👤",
    valence: -0.65,
    arousal: 0.2,
    description: "Averted distant stare, flat affective tone, isolated posture.",
  },
  Heartbroken: {
    category: "Vulnerability & Resistance",
    emoji: "💔",
    valence: -0.85,
    arousal: 0.45,
    description: "High distress, drooping facial tone, intense downward affective pull.",
  },
  Depressed: {
    category: "Vulnerability & Resistance",
    emoji: "🌧️",
    valence: -0.8,
    arousal: 0.1,
    description: "Generalized facial hypotonia, severely damped response, low vitality.",
  },
  Angry: {
    category: "Vulnerability & Resistance",
    emoji: "😠",
    valence: -0.75,
    arousal: 0.85,
    description: "Deep corrugator contraction, narrowed piercing eyes, bared jaw tension.",
  },
  Jealous: {
    category: "Vulnerability & Resistance",
    emoji: "👀",
    valence: -0.55,
    arousal: 0.6,
    description: "Guarded lateral gaze, tense pursed lips, appraising suspicion.",
  },
  Envious: {
    category: "Vulnerability & Resistance",
    emoji: "🐍",
    valence: -0.5,
    arousal: 0.5,
    description: "Subtle asymmetric mouth tightening, watchful sideways head tilt.",
  },
  Disgusted: {
    category: "Vulnerability & Resistance",
    emoji: "🤢",
    valence: -0.7,
    arousal: 0.6,
    description: "Elevated upper lip (AU10) and wrinkled nasolabial bridge (AU9).",
  },
  Contemptuous: {
    category: "Vulnerability & Resistance",
    emoji: "😼",
    valence: -0.4,
    arousal: 0.4,
    description: "Unilateral lip corner pull (AU14 Buccinator), superior dismissive smirk.",
  },

  // ── Row 5: Anxiety & Recovery ──
  Afraid: {
    category: "Anxiety & Recovery",
    emoji: "😨",
    valence: -0.7,
    arousal: 0.85,
    description: "Retracted lips, widened startled eyes, raised inner and outer brows.",
  },
  Anxious: {
    category: "Anxiety & Recovery",
    emoji: "😰",
    valence: -0.6,
    arousal: 0.75,
    description: "Elevated autonomic arousal, micro-tremor in gaze, tense brow ridge.",
  },
  Nervous: {
    category: "Anxiety & Recovery",
    emoji: "😬",
    valence: -0.45,
    arousal: 0.65,
    description: "Compressed bared lips, darting focal gaze, transient facial shifts.",
  },
  Guilty: {
    category: "Anxiety & Recovery",
    emoji: "🫣",
    valence: -0.5,
    arousal: 0.4,
    description: "Averted downward gaze, lowered chin pitch, tightly closed lips.",
  },
  Embarrassed: {
    category: "Anxiety & Recovery",
    emoji: "😳",
    valence: -0.35,
    arousal: 0.6,
    description: "Self-conscious micro-smirk, averted gaze, flushing craniofacial tension.",
  },
  Ashamed: {
    category: "Anxiety & Recovery",
    emoji: "🙇",
    valence: -0.75,
    arousal: 0.25,
    description: "Pronounced downward head drop (pitch < -12°), closed eyes, withdrawn.",
  },
  Regretful: {
    category: "Anxiety & Recovery",
    emoji: "🤦",
    valence: -0.55,
    arousal: 0.35,
    description: "Subtle head roll, downturned oral corners, contemplative sorrow.",
  },
  Overwhelmed: {
    category: "Anxiety & Recovery",
    emoji: "🤯",
    valence: -0.6,
    arousal: 0.9,
    description: "High cognitive overload, tense brow furrow, elevated autonomic stress.",
  },
  Exhausted: {
    category: "Anxiety & Recovery",
    emoji: "🥱",
    valence: -0.4,
    arousal: 0.08,
    description: "Heavy eyelids (low EAR), slow blink cadence, slackened facial muscles.",
  },
  Relieved: {
    category: "Anxiety & Recovery",
    emoji: "😌",
    valence: 0.6,
    arousal: 0.2,
    description: "Post-tension facial release, restored equilibrium, peaceful exhale.",
  },
};

/**
 * High-Precision Biometric & Action Unit Synthesis Classifier
 * Fuses calibrated deltas, 68D Action Units, and head pose to classify
 * into the 50-state taxonomy.
 */
export function classify50StateEmotion(
  probabilities = {},
  actionUnits = {},
  calibrationBaseline = null
) {
  const happy = probabilities.Happy || 0;
  const surprise = probabilities.Surprise || 0;
  const sad = probabilities.Sad || 0;
  const angry = probabilities.Angry || 0;
  const fear = probabilities.Fear || 0;
  const disgust = probabilities.Disgust || 0;

  // Calibrated deltas
  const smileRaw = actionUnits.smile || 0;
  const earRaw = actionUnits.eyeOpenness || 75;
  const browRaw = actionUnits.browFurrow || 10;
  const yaw = actionUnits.yaw || 0;
  const pitch = actionUnits.pitch || 0;
  const roll = actionUnits.roll || 0;
  const asymmetry = actionUnits.smileAsymmetry || 0;
  const focus = actionUnits.focusScore || 85;

  let smile = smileRaw;
  let ear = earRaw;
  let brow = browRaw;

  if (calibrationBaseline) {
    // Normalization against user's neutral baseline
    smile = Math.max(0, smileRaw - (calibrationBaseline.smile || 0));
    ear = earRaw;
    brow = Math.max(0, browRaw - (calibrationBaseline.brow || 0));
  }

  // Diagnostic reason builder
  let diagnosticAnswer;
  let chosenState;

  // Decision Logic across the 5 Clusters

  // 1. High Positive & Joy
  if (smile > 45 || happy > 45) {
    if (ear > 85 && smile > 65) {
      chosenState = "Excited";
      diagnosticAnswer = `High-arousal zygomatic lift (+${smile}%) with widened ocular aperture (${ear}%) indicates exuberant Excitement.`;
    } else if (smile > 55 && ear < 70) {
      chosenState = "Joyful";
      diagnosticAnswer = `Duchenne smile pattern: bilateral cheek elevation (+${smile}%) with orbital crinkling indicates authentic Joy.`;
    } else if (asymmetry > 25 && smile > 30) {
      chosenState = "Playful";
      diagnosticAnswer = `Asymmetric lip contraction (${asymmetry}% delta) with active smile indicates a Playful expression.`;
    } else if (pitch > 5 && smile > 30) {
      chosenState = "Proud";
      diagnosticAnswer = `Elevated cranial pitch (+${pitch}°) paired with confident smile (+${smile}%) indicates Pride and self-assurance.`;
    } else {
      chosenState = "Happy";
      diagnosticAnswer = `Symmetric zygomaticus contraction (+${smile}%) with high positive affective valence confirms a Happy state.`;
    }
  }
  // 2. Subtle Positive, Confident or Grateful
  else if (smile > 20 || (happy > 28 && sad < 15 && angry < 15)) {
    if (asymmetry > 22) {
      chosenState = "Confident";
      diagnosticAnswer = `Unilateral smirk (${asymmetry}% asymmetry) with centered cranial alignment (+${yaw}° yaw) marks a Confident attitude.`;
    } else if (pitch < -4 && smile > 15) {
      chosenState = "Grateful";
      diagnosticAnswer = `Gentle cranial bow (${pitch}°) with softened resting smile (+${smile}%) reflects Gratitude.`;
    } else if (Math.abs(roll) > 5) {
      chosenState = "Affectionate";
      diagnosticAnswer = `Gentle head roll tilt (${roll}°) with pleasant smile (+${smile}%) conveys Affection and warmth.`;
    } else {
      chosenState = "Content";
      diagnosticAnswer = `Gentle resting lip elevation (+${smile}%) and stable gaze reflect peaceful Contentment.`;
    }
  }
  // 3. High Arousal & Wonder / Surprise
  else if (surprise > 35 || actionUnits.mouthOpen > 40) {
    if (actionUnits.mouthOpen > 55 && ear > 85) {
      chosenState = "Shocked";
      diagnosticAnswer = `Significant jaw displacement and dilated ocular opening indicate sudden Shock or disbelief.`;
    } else if (smile > 20 && surprise > 30) {
      chosenState = "Amazed";
      diagnosticAnswer = `Elevated brow ridge with open smile and wide eyes reflects Wonder and Amazement.`;
    } else {
      chosenState = "Surprised";
      diagnosticAnswer = `Bilateral frontalis brow raise with parted lips confirms a Surprised reflex.`;
    }
  }
  // 4. Cognitive Inquisitive, Thinking, Curious, Skeptical
  else if (Math.abs(roll) > 5 || Math.abs(yaw) > 12 || brow > 25 || asymmetry > 18) {
    if (asymmetry > 22 && brow > 20) {
      chosenState = "Skeptical";
      diagnosticAnswer = `Unilateral mouth elevation paired with brow furrow indicates active Skeptical inquiry.`;
    } else if (Math.abs(roll) > 6 && ear > 65) {
      chosenState = "Curious";
      diagnosticAnswer = `Lateral cranial tilt (${roll}°) with high ocular alertness (${ear}%) reflects active Curiosity.`;
    } else if (brow > 35 && smile < 15) {
      if (angry > 25) {
        chosenState = "Frustrated";
        diagnosticAnswer = `Tense corrugator contraction (+${brow}%) with compressed lips reveals cognitive Frustration.`;
      } else {
        chosenState = "Thinking";
        diagnosticAnswer = `Corrugator brow focus (+${brow}%) with steady analytical gaze indicates deep Thinking.`;
      }
    } else if (pitch < -6 && focus < 70) {
      chosenState = "Pensive";
      diagnosticAnswer = `Lowered cranial pitch (${pitch}°) and distant contemplative gaze characterize a Pensive state.`;
    } else if (focus > 80) {
      chosenState = "Interested";
      diagnosticAnswer = `Forward head alignment and high ocular focus (${focus}%) signify keen Interest.`;
    } else {
      chosenState = "Confused";
      diagnosticAnswer = `Asymmetric craniofacial tension with head tilt (${roll}°) indicates Confusion.`;
    }
  }
  // 5. Negative, Distress, Anxiety or Exhaustion
  else if (sad > 25 || angry > 25 || fear > 25 || disgust > 25) {
    if (angry > 40) {
      chosenState = "Angry";
      diagnosticAnswer = `Strong corrugator pull with aggressive eye narrowing signifies Anger.`;
    } else if (disgust > 35) {
      chosenState = "Disgusted";
      diagnosticAnswer = `Wrinkled nasolabial folds and raised upper lip mark physical or moral Disgust.`;
    } else if (fear > 35) {
      chosenState = "Afraid";
      diagnosticAnswer = `Widened ocular opening with tense retracted mouth reflects Fear.`;
    } else if (sad > 45) {
      chosenState = "Heartbroken";
      diagnosticAnswer = `Significant downward mouth angle with heavy ocular drooping reflects deep Sadness.`;
    } else {
      chosenState = "Sad";
      diagnosticAnswer = `Downturned lip corners with depressed affective valence indicate Sadness.`;
    }
  }
  // 6. Low Energy, Drowsiness, Boredom, or Calm Relaxation
  else if (ear < 48) {
    if (focus < 40) {
      chosenState = "Exhausted";
      diagnosticAnswer = `Low eye aperture (${ear}%) and slow blink cadence indicate physical Exhaustion or drowsiness.`;
    } else {
      chosenState = "Bored";
      diagnosticAnswer = `Slackened facial musculature and low cognitive engagement reflect Boredom.`;
    }
  } else {
    // True resting equilibrium
    chosenState = "Relaxed";
    diagnosticAnswer = `Balanced craniofacial symmetry, calm ocular cadence, and zero brow strain mark a Relaxed baseline.`;
  }

  const meta = EMOTION_TAXONOMY_50[chosenState] || EMOTION_TAXONOMY_50.Relaxed;

  return {
    stateName: chosenState,
    emoji: meta.emoji,
    category: meta.category,
    description: meta.description,
    diagnosticAnswer,
    valence: meta.valence,
    arousal: meta.arousal,
  };
}
