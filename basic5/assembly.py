"""Text-preserving alignment assembly and conservative timestamp-based attribution."""
import unicodedata
from .common import require, finite_span, stamp


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower()
        if not c.isspace() and unicodedata.category(c)[0] not in "PSZC")


def restore_text(text, units):
    require(normalized(text) == normalized("".join(u["text"] for u in units)), "Aligner changed transcript content")
    require(units or not text, "Nonempty transcript without alignment units")
    positions, chars = [], []
    for i, char in enumerate(text):
        for c in normalized(char):
            positions.append(i)
            chars.append(c)
    require("".join(chars) == normalized(text), "Unicode mapping is ambiguous")
    consumed, cursor, result = 0, 0, []
    for i, unit in enumerate(units):
        size = len(normalized(unit["text"]))
        require(size > 0, "Aligner returned a punctuation-only unit")
        consumed += size
        require(consumed <= len(positions), "Aligner character count exceeded text")
        if consumed < len(positions):
            require(positions[consumed] != positions[consumed-1], "Alignment splits a normalized Unicode character")
        boundary = positions[consumed] if consumed < len(positions) else len(text)
        if i == len(units)-1:
            boundary = len(text)
        require(boundary > cursor, "Empty mapped text unit")
        result.append(dict(unit, text=text[cursor:boundary], aligner_text=unit["text"]))
        cursor = boundary
    require("".join(u["text"] for u in result) == text, "Text preservation failed")
    return result


def align_or_fallback(chunk, units, duration):
    """Only validation errors become fallbacks. Model/runtime failures belong to the worker."""
    try:
        mapped = restore_text(chunk["text"], units)
        previous = -1.0
        for u in mapped:
            finite_span(u["start"], u["end"], chunk["end"]-chunk["start"], allow_zero=True)
            require(u["start"] >= previous, "Nonmonotonic aligned units")
            previous = u["end"]
        words = [dict(u, start=u["start"]+chunk["start"], end=u["end"]+chunk["start"],
            chunk_id=chunk["id"], alignment_status="aligned", timestamp_source="qwen_sdk_postprocessed") for u in mapped]
        for word in words:
            finite_span(word["start"],word["end"],duration,allow_zero=True)
        return words, None
    except (ValueError, KeyError, TypeError) as error:
        return fallback(chunk, str(error)), str(error)


def fallback(chunk, reason):
    if not chunk["text"]:
        return []
    return [dict(start=chunk["start"], end=chunk["end"], text=chunk["text"], chunk_id=chunk["id"],
        alignment_status="unaligned_text_preserved", timestamp_source="chunk_bounds", alignment_error=reason)]


def _union_duration(spans):
    end, total = None, 0.0
    for a,b in sorted(spans):
        if end is None or a > end:
            total += b-a
        elif b > end:
            total += b-end
        end = b if end is None else max(end,b)
    return total


def assign_words(words, turns, cfg):
    output = []
    for word in words:
        a,b = word["start"],word["end"]
        spans = {}
        for turn in turns:
            if a == b:
                if turn["start"] <= a < turn["end"]:
                    spans.setdefault(turn["speaker"], []).append((a,b))
            else:
                lo,hi = max(a,turn["start"]),min(b,turn["end"])
                if hi>lo:
                    spans.setdefault(turn["speaker"], []).append((lo,hi))
        scores = sorted(((speaker, _union_duration(values)/(b-a) if b>a else 1.0)
            for speaker,values in spans.items()),key=lambda pair:(-pair[1],pair[0]))
        ambiguous = len(scores)>1 and (scores[1][1]>=cfg["min_word_speaker_coverage"] or scores[0][1]-scores[1][1]<cfg["speaker_ambiguity_margin"])
        # Chunk fallback is too coarse to claim a word-level speaker decision.
        confident = bool(scores) and scores[0][1]>=cfg["min_word_speaker_coverage"] and not ambiguous and word["alignment_status"]=="aligned"
        output.append(dict(word, speaker=scores[0][0] if confident else "unknown",
            speaker_candidates=[dict(speaker=s, coverage=v if b>a else None) for s,v in scores],
            speaker_uncertainty=None if confident else "ambiguous_or_insufficient_activity_or_alignment"))
    return output


def assemble(chunks, aligned, turns, cfg, duration):
    require([x["id"] for x in chunks] == [x["id"] for x in aligned], "Alignment chunk inventory differs")
    for turn in turns:
        finite_span(turn["start"],turn["end"],duration)
    words, failures = [], []
    for chunk, result in zip(chunks, aligned):
        require(result["text"]==chunk["text"], "Alignment transcript text changed")
        require("".join(w["text"] for w in result["words"])==chunk["text"], "Alignment lost text")
        for word in result["words"]:
            finite_span(word["start"],word["end"],duration,allow_zero=True)
        words.extend(result["words"])
        if result.get("error"):
            failures.append(dict(chunk_id=chunk["id"],reason=result["error"]))
    words = assign_words(words,turns,cfg)
    segments = []
    for word in words:
        if (segments and segments[-1]["speaker"]==word["speaker"] and
                segments[-1]["chunk_id"]==word["chunk_id"] and word["start"]-segments[-1]["end"]<=1):
            segments[-1]["text"] += word["text"]
            segments[-1]["end"] = max(segments[-1]["end"],word["end"])
        else:
            segments.append({k:word[k] for k in ("start","end","speaker","text","chunk_id")})
    require("".join(s["text"] for s in segments)=="".join(c["text"] for c in chunks), "Final transcript lost or changed text")
    return dict(segments=segments,words=words,speaker_activity=turns,
        quality_flags=dict(alignment_fallback_chunks=failures,
            zero_duration_units=sum(w["start"]==w["end"] for w in words),
            unknown_speaker_units=sum(w["speaker"]=="unknown" for w in words),
            overlap_separation=False,word_timing_accuracy_validated=False))


def export_text(result):
    return "\n".join(f"[{stamp(s['start'])} --> {stamp(s['end'])}] {s['speaker']}：{s['text']}" for s in result["segments"])+"\n"


def export_srt(result):
    cues=[]
    for s in result["segments"]:
        # Keep zero-duration content in JSON/TXT. SRT cannot represent it honestly.
        if round(s["end"]*1000)<=round(s["start"]*1000):
            continue
        cues.append(f"{len(cues)+1}\n{stamp(s['start'], ',')} --> {stamp(s['end'], ',')}\n[{s['speaker']}] {s['text']}\n")
    return "\n".join(cues)
