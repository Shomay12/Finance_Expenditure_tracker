# LLM Agent Module: Workflow & Developer Process Guide

---

## 1. Quickstart & Execution Workflows

### 1. Interactive Terminal Chat Console
Launch an interactive session for any user:
```bash
python3 LLM_Agent/cli.py --interactive --user USR_0001
```

### 2. Run Automated 8-Scenario Benchmark
Evaluate all 8 financial scenarios (Breakdown, Month-to-month, Anomalies, Recurring/SIP, Forecast, Behavior, Missing Data, Injection Defense):
```bash
python3 LLM_Agent/cli.py --demo
```

### 3. Run Automated Unit & Security Tests
Verify user isolation, prompt injection defense, Groq connectivity, and uncertainty communication:
```bash
python3 -m unittest discover -s LLM_Agent/tests
```

### 4. Test Groq Cloud Connectivity
```bash
python3 LLM_Agent/cli.py --test-connection
```

---

## 2. Developer Workflow & Adding Features

1. **Context Extensions**: Modify `LLM_Agent/context_builder.py` to add new structured fields (ensure `FinancialContextBuilder.sanitize_for_json` is applied).
2. **Model Upgrades**: Update `LLM_Agent/config.py` when adding new Groq model identifiers.
3. **Guardrails**: Modify `LLM_Agent/prompts/system_prompt.py` for specialized policy adjustments while maintaining the 24 invariants.
