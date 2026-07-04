"""Message scaffold, bare-refusal detector, completion cleanup (no GPU)."""

from rewriter.rewrite import (
    REFUSE_CONTENT_LABEL,
    REFUSE_USER_SUFFIX,
    REWRITE_RETURN_INSTRUCTION,
    REWRITE_SAFETY_SUFFIX,
    UNSAFE_DRAFT_LABEL,
    build_messages,
    build_refusal_retry_messages,
    clean_completion,
    is_bare_refusal,
)

CONTENT = "some flagged response text"
USER = "the original user request"
SYSTEM = "the selected prompt pack"


class TestBuildMessages:
    def test_rewrite_scaffold(self):
        msgs = build_messages(CONTENT, USER, "REWRITE", SYSTEM)
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == SYSTEM
        user = msgs[1]["content"]
        # exact scaffold, in order
        assert user == "\n".join([
            UNSAFE_DRAFT_LABEL,
            CONTENT,
            "\nOriginal user request:\n" + USER,
            REWRITE_RETURN_INSTRUCTION,
            REWRITE_SAFETY_SUFFIX,
        ])

    def test_refuse_scaffold(self):
        msgs = build_messages(CONTENT, USER, "REFUSE", SYSTEM)
        user = msgs[1]["content"]
        assert user == "\n".join([
            REFUSE_CONTENT_LABEL,
            CONTENT,
            "\nOriginal user request:\n" + USER,
            REFUSE_USER_SUFFIX,
        ])

    def test_decision_case_insensitive(self):
        assert build_messages(CONTENT, USER, " refuse ", SYSTEM)[1]["content"].startswith(
            REFUSE_CONTENT_LABEL
        )

    def test_no_user_input(self):
        user = build_messages(CONTENT, None, "REWRITE", SYSTEM)[1]["content"]
        assert "Original user request" not in user

    def test_retry_messages(self):
        msgs = build_refusal_retry_messages("x" * 2000, USER, "REWRITE")
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert "produced a bare refusal" in msgs[0]["content"]
        assert "...[truncated]" in msgs[1]["content"]  # 1500-char preview cap
        assert msgs[1]["content"].endswith(REFUSE_USER_SUFFIX)


class TestBareRefusalDetector:
    def test_bare_one_liners(self):
        assert is_bare_refusal("I'm sorry, but I can't assist with that request.")
        assert is_bare_refusal("I cannot help with this.")
        assert is_bare_refusal("很抱歉，我无法帮助。")
        assert is_bare_refusal("Lo siento, no puedo ayudar con eso.")

    def test_contextual_refusal_is_not_bare(self):
        assert not is_bare_refusal(
            "I can't help with that because it involves explicit content. "
            "You might instead look for general relationship-communication resources."
        )

    def test_benign_openers_are_not_refusals(self):
        assert not is_bare_refusal("The weather is nice today.")
        assert not is_bare_refusal("")

    def test_long_single_sentence_is_not_bare(self):
        assert not is_bare_refusal(
            "I'm sorry, but I can't assist with that request because " + "x" * 80
        )

    def test_emphatic_punctuation_counts_once(self):
        assert is_bare_refusal("I'm sorry, I can't help!!!")


class TestCleanCompletion:
    def test_role_marker_content_preserved(self):
        # With the corrected decode slice there is no prompt echo to rescue;
        # a rewrite that legitimately contains a role-marker string must
        # survive intact (review finding #1).
        text = "safe transcript rewrite:\nassistant\nplease be kind"
        assert clean_completion(text) == text

    def test_prefix_strip(self):
        assert clean_completion("Rewritten response: safe text") == "safe text"

    def test_plain_text_untouched(self):
        assert clean_completion("just a normal rewrite") == "just a normal rewrite"
