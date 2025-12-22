# ============================================================
# FIXED CLEANUP SCRIPT - Extracts Actual Questions
# Now with complete Turkish punctuation support
# ============================================================

import json
import re
from pathlib import Path

def load_raw_file(path):
    """Load the full dataset from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_output(path, data):
    """Write cleaned data back to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def preview(text, limit=120):
    """Utility: return a shortened, safe preview for logs."""
    if not isinstance(text, str):
        return str(text)
    return text if len(text) <= limit else text[:limit] + "..."

def extract_actual_content(text, is_question=True):
    """
    Extract actual question or complete answer from model's rambling.
    
    For QUESTIONS: Tries to find the actual technical question being asked.
    For ANSWERS: Just ensures complete sentences.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    
    # Step 1: Normalize whitespace
    normalized = re.sub(r"\s+", " ", text).strip()
    
    # Define ALL Turkish sentence endings
    # Note: We include the space after punctuation for better matching
    TURKISH_ENDINGS = (".", "?", "!", "…", ".”", "?”", "!”", ":", ";")
    
    if is_question:
        # ============================================================
        # STRATEGY FOR QUESTIONS: Find the actual technical question
        # ============================================================
        
        # Pattern 1: Direct question with any Turkish question mark
        # Look for ? or ?” or ?'
        question_marks_pattern = r'[?](?:[”\'])?'
        question_matches = list(re.finditer(question_marks_pattern, normalized))
        
        if question_matches:
            # Take everything up to the last question mark
            last_qm = question_matches[-1]
            candidate = normalized[:last_qm.end()].strip()
            
            # Validate: is this really a question or more meta-commentary?
            # Real questions usually don't start with "Hmm..." or talk about users
            if not re.match(r'^(Hmm|Tamam|Kullanıcı|Önceki|Şimdi|Demek ki|Yani|Peki)', candidate, re.IGNORECASE):
                # Additional check: Question should be at least 3 words
                if len(candidate.split()) >= 3:
                    return candidate
        
        # Pattern 2: Look for Turkish question words with proper endings
        # Turkish question words that should end sentences
        question_patterns = [
            r'(?:nedir|nasıl|ne zaman|nerede|kim|hangi|kaç|niçin|niye|nasıldır)[?!”]?\s*$',  # Question words
            r'(?:açıklayınız|açıklayın|tanımlayınız|tanımlayın|anlatınız|anlatın)[.!?…”]?\s*$',  # Imperatives
            r'(?:soru(?:su|muz)?\s*:)\s*(.+?)(?:\.\s|$)',  # "soru:" pattern
            r'(?:soruyorum|soruyoruz|sorulur|sorulmuştur)[.!?…”]?\s*$',  # Question statements
        ]
        
        for pattern in question_patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                if ":" in pattern:
                    # Extract text after colon
                    extracted = match.group(1).strip()
                    # Ensure it has a proper ending
                    if extracted and not any(extracted.endswith(e) for e in TURKISH_ENDINGS):
                        extracted += "?"
                    return extracted
                else:
                    # Return the matched question with context
                    # Find start of this sentence
                    start = 0
                    for ending in TURKISH_ENDINGS:
                        # Look for previous sentence ending
                        prev_end = normalized.rfind(ending + " ", 0, match.start())
                        if prev_end > start:
                            start = prev_end + len(ending) + 1
                    
                    result = normalized[start:match.end()].strip()
                    # Ensure question mark if it's a question word
                    if any(qword in result.lower() for qword in ['nedir', 'nasıl', 'ne zaman', 'nerede']):
                        if not result.endswith("?"):
                            result += "?"
                    return result
        
        # Pattern 3: Extract topic and create simple question
        # Find topic in quotes or after "konusunda/hakkında"
        topic = None
        
        # Try to find topic in quotes: "Veritabanı Tasarımı" or 'Veritabanı Tasarımı'
        quote_match = re.search(r'["\']([^"\']+)["\']', normalized)
        if quote_match:
            topic = quote_match.group(1)
        else:
            # Find topic after "konusunda" or "hakkında"
            topic_match = re.search(r'(?:konusunda|hakkında)[^.!?…]*?(\b[\w\s]+\b)(?=[.!?…]|$)', normalized)
            if topic_match:
                topic = topic_match.group(1).strip()
        
        if topic:
            # Create a simple technical question about the topic
            question_formats = [
                f"{topic} nedir?",
                f"{topic} nasıl çalışır?",
                f"{topic}'nın temel prensipleri nelerdir?",
                f"{topic} hakkında teknik bir soru nedir?"
            ]
            return question_formats[0]  # Use first format
        
        # Pattern 4: Find any complete sentence ending with proper Turkish punctuation
        # Split by Turkish sentence endings
        sentences = re.split(r'[.!?…](?:[”\'])?\s+', normalized)
        for sentence in reversed(sentences):  # Check from end (most recent)
            sentence = sentence.strip()
            if len(sentence) > 15:  # Reasonable length
                # Check if it looks like a technical question/statement
                is_technical = any(word in sentence.lower() for word in [
                    'veri', 'algoritma', 'sistem', 'program', 'kod', 'teknik',
                    'yapı', 'mimar', 'tasarım', 'geliştir', 'üret', 'analiz'
                ])
                is_meta = re.match(r'^(Hmm|Tamam|Kullanıcı|Önceki|Şimdi|Demek ki|Yani|Peki|Anladım)', sentence, re.IGNORECASE)
                
                if is_technical and not is_meta:
                    # Add appropriate ending
                    if not any(sentence.endswith(e) for e in TURKISH_ENDINGS):
                        # If it has question words, add ? otherwise .
                        if any(qword in sentence.lower() for qword in ['nedir', 'nasıl', 'ne zaman', 'nerede']):
                            sentence += "?"
                        else:
                            sentence += "."
                    return sentence
        
        # Fallback: First complete sentence without meta-commentary
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not re.match(r'^(Hmm|Tamam|Kullanıcı|Önceki)', sentence, re.IGNORECASE):
                if not any(sentence.endswith(e) for e in TURKISH_ENDINGS):
                    sentence += "."
                return sentence
    
    else:
        # ============================================================
        # STRATEGY FOR ANSWERS: Ensure complete sentences with Turkish endings
        # ============================================================
        
        # Find last complete sentence with proper Turkish ending
        # Create regex that matches all Turkish endings
        endings_regex = r'[.!?…](?:[”\'])?(?:\s|$)'
        endpoints = list(re.finditer(endings_regex, normalized))
        
        if endpoints:
            last_end = endpoints[-1].end()
            result = normalized[:last_end].strip()
            
            # Special case: If we cut at an ellipsis and there's more content
            if result.endswith("…") and len(result) < len(normalized):
                # Check if next character starts a new thought
                next_char = normalized[last_end:last_end+5].strip()
                if next_char and not re.match(r'^(ve|ama|fakat|ancak|çünkü)', next_char, re.IGNORECASE):
                    # Ellipsis might be intentional (trailing off)
                    return result
                # Otherwise, ellipsis might be mid-sentence
                # Try to find a better break point
                pass
            
            return result
        
        # If no proper endings, try to complete at a logical break
        logical_breaks = list(re.finditer(r'(?:[,-]\s|:\s|\b(?:ancak|fakat|oysa|çünkü|böylece|sonuç|örnek|örneğin)\b)', normalized))
        if logical_breaks:
            last_break = logical_breaks[-1].end()
            # Only cut if there's substantial text (at least 20 chars) after the break
            if len(normalized) - last_break > 20:
                result = normalized[:last_break].strip()
                # Add ellipsis to indicate continuation
                if not result.endswith("…"):
                    result += "…"
                return result
    
    # If all else fails, return original but cleaned up
    # Ensure it ends with proper punctuation
    if not any(normalized.endswith(e) for e in TURKISH_ENDINGS):
        # For questions, add ?; for statements, add .
        if is_question and any(qword in normalized.lower() for qword in ['nedir', 'nasıl', 'ne zaman', 'nerede', 'soru']):
            normalized += "?"
        else:
            normalized += "."
    
    return normalized

def main():
    input_path = Path("raw_qa_data.json")
    output_path = Path("cleaned_qa_data.json")
    log_path = Path("cleanup_log.txt")
    
    print(f"Loading dataset from: {input_path}")
    data = load_raw_file(input_path)
    
    cleaned = []
    log_lines = []
    question_fixes = 0
    answer_fixes = 0
    
    print(f"Processing {len(data)} records...")
    
    for idx, item in enumerate(data):
        q_orig = item.get("question", "")
        a_orig = item.get("answer", "")
        
        # Extract actual question (special handling)
        q_fixed = extract_actual_content(q_orig, is_question=True)
        
        # Clean answer (just ensure complete sentences)
        a_fixed = extract_actual_content(a_orig, is_question=False)
        
        # Log changes
        if q_fixed != q_orig:
            question_fixes += 1
            log_lines.append(
                f"[QUESTION FIXED] Index: {idx} (Label: {item.get('label')}, Set: {item.get('set_type')})\n"
                f"   ORIGINAL ({len(q_orig)} chars): {preview(q_orig, 120)}\n"
                f"   FIXED    ({len(q_fixed)} chars): {preview(q_fixed, 120)}\n"
                f"{'-'*80}\n"
            )
        
        if a_fixed != a_orig:
            answer_fixes += 1
            # Only log answer fixes if they're substantial (not just adding punctuation)
            if len(a_fixed) < len(a_orig) * 0.95 or len(a_fixed) > len(a_orig) * 1.05:
                log_lines.append(
                    f"[ANSWER FIXED] Index: {idx} (Label: {item.get('label')}, Set: {item.get('set_type')})\n"
                    f"   ORIGINAL ({len(a_orig)} chars): {preview(a_orig, 100)}\n"
                    f"   FIXED    ({len(a_fixed)} chars): {preview(a_fixed, 100)}\n"
                    f"{'-'*80}\n"
                )
        
        cleaned.append({
            "set_type": item.get("set_type"),
            "question": q_fixed,
            "answer": a_fixed,
            "label": item.get("label"),
        })
    
    # Write cleaned data
    write_output(output_path, cleaned)
    
    # Write log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"CLEANUP REPORT - Turkish Punctuation Aware\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total records processed: {len(cleaned)}\n")
        f.write(f"Questions fixed: {question_fixes} ({question_fixes/len(cleaned)*100:.1f}%)\n")
        f.write(f"Answers fixed: {answer_fixes} ({answer_fixes/len(cleaned)*100:.1f}%)\n\n")
        f.write(f"DETAILED CHANGES:\n")
        f.write(f"{'='*60}\n\n")
        f.writelines(log_lines)
    
    # Quality analysis
    print(f"\n{'='*60}")
    print("CLEANUP COMPLETE - Turkish Punctuation Aware")
    print(f"{'='*60}")
    print(f"Total records: {len(cleaned)}")
    print(f"Questions fixed: {question_fixes} ({question_fixes/len(cleaned)*100:.1f}%)")
    print(f"Answers fixed: {answer_fixes} ({answer_fixes/len(cleaned)*100:.1f}%)")
    
    # Check final quality
    TURKISH_ENDINGS = (".", "?", "!", "…", ".”", "?”", "!”", ":", ";")
    
    questions_with_proper_end = sum(
        1 for item in cleaned 
        if any(item["question"].endswith(e) for e in TURKISH_ENDINGS)
    )
    
    questions_with_qmark = sum(1 for item in cleaned if "?" in item["question"])
    
    print(f"\nQUALITY METRICS:")
    print(f"Questions with proper Turkish ending: {questions_with_proper_end}/{len(cleaned)} ({questions_with_proper_end/len(cleaned)*100:.1f}%)")
    print(f"Questions containing '?': {questions_with_qmark}/{len(cleaned)} ({questions_with_qmark/len(cleaned)*100:.1f}%)")
    
    # Average lengths
    avg_q_len = sum(len(item["question"]) for item in cleaned) / len(cleaned)
    avg_a_len = sum(len(item["answer"]) for item in cleaned) / len(cleaned)
    
    print(f"\nAVERAGE LENGTHS:")
    print(f"Questions: {avg_q_len:.1f} characters")
    print(f"Answers: {avg_a_len:.1f} characters")
    
    # Sample cleaned questions
    print(f"\nSAMPLE CLEANED QUESTIONS (first 5 unique):")
    seen = set()
    sample_count = 0
    for i, item in enumerate(cleaned):
        q = item["question"]
        if q not in seen and sample_count < 5:
            seen.add(q)
            print(f"  {sample_count+1}. {preview(q, 100)}")
            sample_count += 1
    
    print(f"\nOutput saved to: {output_path}")
    print(f"Log saved to: {log_path}")

if __name__ == "__main__":
    main()