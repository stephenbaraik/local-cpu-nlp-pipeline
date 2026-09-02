from __future__ import annotations

from pipeline.config import Config
from pipeline.stages import DocContext
from pipeline.stages.clean import CleanStage
from pipeline.stages.validate import ValidateStage

CONFIG = Config()


def _clean(pages: list[str]) -> dict:
    doc = DocContext(
        doc_id="test",
        pdf_bytes=b"",
        config=CONFIG,
        payloads={"extract": {"pages": pages}},
    )
    return CleanStage().run(doc)


def _validate(body: str) -> dict:
    doc = DocContext(
        doc_id="test",
        pdf_bytes=b"",
        config=CONFIG,
        payloads={"clean": {"body": body}},
    )
    return ValidateStage().run(doc)


ARTICLE_PARAGRAPH = (
    "Researchers published a detailed study this week describing how attackers are "
    "exploiting misconfigured cloud storage buckets to exfiltrate sensitive data at "
    "scale. The report, compiled over six months of incident response engagements, "
    "found that nearly a third of breaches traced back to publicly readable storage "
    "buckets left open by default settings. Security teams are urged to audit their "
    "storage configurations regularly and enable logging on every bucket, the authors "
    "wrote, adding that automated scanning tools can catch most of these "
    "misconfigurations before they are exploited by opportunistic attackers scanning "
    "the internet for open buckets."
)

NAV_BLOCK = "Home | About | Contact | Blog | Careers | Support"
COOKIE_BLOCK = "We use cookies to improve your experience. Accept All Cookies to continue browsing this site."
NEWSLETTER_BLOCK = "Sign up for our Newsletter\nGet weekly updates delivered to your inbox. Subscribe now."
TITLE_BLOCK = "This is the real article headline"
RELATED_BLOCK = (
    "Related Articles\n"
    "Ransomware gangs shift tactics\n"
    "Zero-day patched in popular router firmware\n"
    "Cloud misconfiguration causes major leak"
)
FOLLOW_BLOCK = "Follow us on social media for more updates. All rights reserved."


def test_clean_removes_boilerplate_keeps_article_body():
    page = "\n\n".join(
        [NAV_BLOCK, COOKIE_BLOCK, NEWSLETTER_BLOCK, TITLE_BLOCK, ARTICLE_PARAGRAPH, RELATED_BLOCK, FOLLOW_BLOCK]
    )
    result = _clean([page])
    body = result["body"]

    assert "exfiltrate sensitive data" in body
    for boilerplate in ("Careers", "Accept All Cookies", "Subscribe now", "Ransomware gangs", "Follow us"):
        assert boilerplate not in body


def test_clean_removes_repeated_headers_across_pages():
    header = "CyberDaily Weekly Briefing"
    footer = "© 2026 CyberDaily. All rights reserved."
    page1 = "\n\n".join([header, ARTICLE_PARAGRAPH, footer])
    page2 = "\n\n".join([header, ARTICLE_PARAGRAPH.replace("Researchers", "Analysts"), footer])
    page3 = "\n\n".join([header, ARTICLE_PARAGRAPH.replace("this week", "last month"), footer])

    result = _clean([page1, page2, page3])

    assert header in result["removed_lines"]
    assert header not in result["body"]
    assert footer not in result["body"]
    assert "exfiltrate sensitive data" in result["body"]


def test_short_genuine_article_is_accepted():
    genuine = " ".join([ARTICLE_PARAGRAPH] * 2)  # comfortably clears the 120-word floor
    result = _validate(genuine)
    assert result["status"] == "accepted"
    assert result["reason"] is None
    assert result["signals"]["failure_signal"] is None


CLOUDFLARE_INTERSTITIAL = (
    "Attention Required! | Cloudflare. Sorry, you have been blocked. You are unable to "
    "access this website because we believe you are using automation tools to browse "
    "the site. This behavior may be caused by a browser extension, or a script that "
    "sends automated requests. Please verify you are a human by completing the "
    "challenge below. Ray ID: 84a2f3c9d1234567. IP: 203.0.113.42. Performance and "
    "security by a content delivery network."
)


def test_cloudflare_interstitial_is_rejected_by_rule_one():
    cleaned = _clean([CLOUDFLARE_INTERSTITIAL])
    result = _validate(cleaned["body"])
    assert result["status"] == "rejected"
    assert result["reason"] == "failure_signal"
    assert result["signals"]["failure_signal"] is not None


def test_padded_403_page_is_rejected_by_rule_two_not_rule_one():
    sentence = "Error 403 Forbidden access denied please contact the site administrator for assistance."
    padded = " ".join([sentence] * 20)  # far past the 120-word floor, but almost no unique words

    cleaned = _clean([padded])
    result = _validate(cleaned["body"])

    assert result["signals"]["content_words"] > 120
    assert result["status"] == "rejected"
    assert result["reason"] == "low_unique_word_ratio"
