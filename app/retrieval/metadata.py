import os
import re

# ── Tag Taxonomy (aligned to Credit Risk FAQ PDF) ───────────────────────────

_LOAN_PRODUCT_RULES = {
    "home_loan":     r"\b(home loan|property|LTV|mortgage|under.?construction|RERA)\b",
    "business_loan": r"\b(business loan|MSME|GST|turnover|working capital|startup)\b",
    "vehicle_loan":  r"\b(vehicle loan|car loan|auto loan|two.?wheeler)\b",
    "personal_loan": r"\b(personal loan|unsecured|salaried|salary|consumer)\b",
}

_RISK_FACTOR_RULES = {
    "cibil":           r"\b(CIBIL|credit score|credit history|bureau|TransUnion)\b",
    "dti":             r"\b(DTI|debt.?to.?income|EMI|monthly obligation|repayment capacity)\b",
    "collateral":      r"\b(collateral|security|pledge|hypothecation|mortgage|LTV)\b",
    "default_history": r"\b(default|NPA|delinquency|overdue|written.?off|settled|restructur)\b",
    "employment":      r"\b(employment|salaried|self.?employed|freelancer|tenure|job|income stability)\b",
    "ltv":             r"\b(LTV|loan.?to.?value|property value|disbursement|RBI cap)\b",
    "income":          r"\b(income|salary|ITR|gross|net income|bank statement|rental income)\b",
}

_POLICY_CATEGORY_RULES = {
    "rbi_compliance":      r"\b(RBI|Reserve Bank|circular|regulation|compliance|guideline|KYC|IRDAI|SEBI)\b",
    "risk_classification": r"\b(low risk|moderate risk|high risk|risk score|classification|flag)\b",
    "documentation":       r"\b(document|salary slip|Form 16|bank statement|Aadhaar|PAN|ITR|proof)\b",
    "pricing":             r"\b(interest rate|pricing|risk.?based|spread|yield|rate grid)\b",
    "eligibility":         r"\b(eligible|qualify|threshold|minimum|maximum|criteria|approve|reject)\b",
}

_CONTENT_TYPE_RULES = {
    "formula":        r"\b(formula|calculate|calculation|ratio|percentage|÷|×|DTI =|LTV =)\b",
    "case_precedent": r"\b(case|precedent|past|historical|example|borrower profile|similar)\b",
    "guideline":      r"\b(guideline|policy|norm|standard|rule|mandate|requirement|RBI)\b",
    "policy":         r"\b(approve|reject|sanction|condition|override|escalate|committee)\b",
}


def _match_tags(text: str, rules: dict) -> list[str]:
    """Return all tag keys whose regex pattern matches the chunk text."""
    matched = []
    for tag, pattern in rules.items():
        if re.search(pattern, text, re.IGNORECASE):
            matched.append(tag)
    return matched or ["general"]


def _tag_chunk(text: str) -> dict:
    """Run all rule sets against a chunk and return the metadata tag dict."""
    return {
        "loan_product":      _match_tags(text, _LOAN_PRODUCT_RULES),
        "risk_factor":       _match_tags(text, _RISK_FACTOR_RULES),
        "policy_category":   _match_tags(text, _POLICY_CATEGORY_RULES),
        "content_type":      _match_tags(text, _CONTENT_TYPE_RULES),
    }


# ── Public enrichment function ──────────────────────────────────────────────
def enrich_metadata(docs, file_path: str):
    """
    Enrich each document chunk with:
      - source, document_extension, page, last_updated  (original fields)
      - loan_product, risk_factor, policy_category, content_type  (new tags)
    """
    last_updated = os.path.getmtime(file_path)

    for doc in docs:
        tags = _tag_chunk(doc.page_content)

        doc.metadata.update({
            # ── original fields ──────────────────────────────────────────
            "source":               file_path,
            "document_extension":   "pdf",
            "page":                 doc.metadata.get("page"),
            "last_updated":         last_updated,
            # ── new classification tags ──────────────────────────────────
            "loan_product":         tags["loan_product"],
            "risk_factor":          tags["risk_factor"],
            "policy_category":      tags["policy_category"],
            "content_type":         tags["content_type"],
        })

    return docs
