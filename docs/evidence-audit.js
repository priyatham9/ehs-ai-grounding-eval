/* A read-only view of the deterministic, versioned audit. No browser-side scoring. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const escape = (s) =>
    String(s ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const pct = (n) => `${Math.round(n * 100)}%`;
  const human = (s) => s.replace(/_/g, " ");
  const state = {
    filter: "all",
    query: "",
    selected: "loto-001",
    mode: "own",
    comparePair: "metadata",
    example: null,
  };
  const filterNames = {
    all: "all factual questions",
    baseline: "BM25 lexical passes",
    recoverable: "recoverable selection misses",
    unreachable: "questions with no passing excerpt",
    "own-fails": "BM25 selected the attached source, but failed",
    metadata: "questions whose attainable pass changes without metadata",
  };
  let audit, items, byId, excerpts, baseline, baselineById, toastTimer;

  function announce(message) {
    $("live-status").textContent = message;
  }
  function toast(message) {
    $("toast").textContent = message;
    $("toast").hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      $("toast").hidden = true;
    }, 3200);
  }
  function dot(pass) {
    return `<span class="status-dot ${pass ? "pass" : "fail"}" aria-label="${pass ? "Lexical pass" : "No lexical pass"}" role="img"></span>`;
  }
  function isMetadata(item) {
    return (
      Boolean(item.passing_excerpt_ids.length) !==
      Boolean(item.passing_bare_excerpt_ids.length)
    );
  }
  function matchesFilter(item) {
    const b = baselineById.get(item.item_id);
    if (state.filter === "baseline") return b.passes;
    if (state.filter === "recoverable")
      return !b.passes && item.passing_excerpt_ids.length > 0;
    if (state.filter === "unreachable")
      return item.passing_excerpt_ids.length === 0;
    if (state.filter === "own-fails") return b.self_match && !b.passes;
    if (state.filter === "metadata") return isMetadata(item);
    return true;
  }
  function visibleItems() {
    const q = state.query.trim().toLowerCase();
    return items.filter(
      (item) =>
        matchesFilter(item) &&
        (!q ||
          [
            item.item_id,
            item.question,
            item.domain,
            item.source_clause,
            item.source_title,
            item.family,
          ]
            .join(" ")
            .replace(/_/g, " ")
            .toLowerCase()
            .includes(q)),
    );
  }
  function writeUrl() {
    const params = new URLSearchParams({
      question: state.selected,
      view: state.mode,
    });
    if (state.example) params.set("excerpt", state.example);
    if (state.mode === "compare") params.set("compare", state.comparePair);
    if (state.filter !== "all") params.set("filter", state.filter);
    if (state.query) params.set("q", state.query);
    history.replaceState(
      null,
      "",
      `${location.pathname}${location.search}#${params}`,
    );
  }
  function readUrl() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (byId.has(p.get("question"))) state.selected = p.get("question");
    if (["bm25", "own", "best", "bare", "compare"].includes(p.get("view")))
      state.mode = p.get("view");
    if (["metadata", "ranking"].includes(p.get("compare")))
      state.comparePair = p.get("compare");
    if (Object.hasOwn(filterNames, p.get("filter")))
      state.filter = p.get("filter");
    state.query = p.get("q") || "";
    const example = audit.score_validity_examples.find(
      (e) =>
        e.question_id === state.selected && e.excerpt_id === p.get("excerpt"),
    );
    state.example = example ? example.excerpt_id : null;
    if (state.example) state.mode = "best";
    $("question-search").value = state.query;
  }
  function renderLedger() {
    const rows = visibleItems();
    $("question-rows").innerHTML = rows
      .map((item) => {
        const b = baselineById.get(item.item_id);
        return `<tr class="${state.selected === item.item_id ? "selected" : ""}" data-row="${escape(item.item_id)}"><td><button class="question-select" data-question="${escape(item.item_id)}" aria-label="Inspect ${escape(item.item_id)}: ${escape(item.question)}" ${state.selected === item.item_id ? 'aria-current="true"' : ""}><span class="row-id">${escape(item.item_id)}${isMetadata(item) ? '<span class="row-meta-warning" aria-label="Metadata-sensitive">△</span>' : ""}</span><span class="row-topic">${escape(item.question)}</span></button></td><td>${dot(b.passes)}</td><td>${dot(item.own_wrapped.passes)}</td><td>${dot(item.passing_excerpt_ids.length > 0)}</td><td>${dot(item.passing_bare_excerpt_ids.length > 0)}</td></tr>`;
      })
      .join("");
    $("visible-count").textContent = `${rows.length} / ${items.length}`;
    $("empty-state").hidden = rows.length !== 0;
    $("filter-summary").textContent = `Showing ${filterNames[state.filter]}`;
    document.querySelectorAll("[data-filter]").forEach((button) => {
      const active = button.dataset.filter === state.filter;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    const selectedRow = $("question-rows").querySelector("tr.selected");
    if (selectedRow) {
      const scroller = document.querySelector(".ledger-scroll");
      const top =
        selectedRow.offsetTop - scroller.querySelector("thead").offsetHeight;
      if (
        top < scroller.scrollTop ||
        top + selectedRow.offsetHeight >
          scroller.scrollTop + scroller.clientHeight
      )
        scroller.scrollTop = top;
    }
    return rows;
  }
  function setFilter(filter, jump = false) {
    state.filter = filter;
    state.example = null;
    const rows = visibleItems();
    if (rows.length && !rows.some((i) => i.item_id === state.selected))
      state.selected = rows[0].item_id;
    renderLedger();
    renderInspector();
    writeUrl();
    announce(`${rows.length} questions. ${filterNames[filter]}.`);
    if (jump)
      $("experiment").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function selectQuestion(id, options = {}) {
    if (!byId.has(id)) return;
    const restoreFocus = document.activeElement?.dataset.question === id;
    state.selected = id;
    state.example = options.example || null;
    if (options.mode) state.mode = options.mode;
    if (options.resetFilter) {
      state.filter = "all";
      state.query = "";
      $("question-search").value = "";
    }
    renderLedger();
    renderInspector();
    if (restoreFocus)
      $("question-rows")
        .querySelector(`[data-question="${id}"]`)
        .focus({ preventScroll: true });
    writeUrl();
    announce(`Selected ${id}. ${byId.get(id).question}`);
    if (options.scroll || matchMedia("(max-width: 800px)").matches) {
      $("inspector").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }
  function phrasePattern(term, flags = "gi") {
    const body = term
      .normalize("NFKC")
      .replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
      .trim()
      .replace(/\s+/g, "\\s+");
    return new RegExp(
      `(?<![a-z0-9])${body}${/\d$/.test(term) ? "(?![0-9])" : ""}`,
      flags,
    );
  }
  function hasLiteral(text, group) {
    return group.some((term) => phrasePattern(term, "i").test(text));
  }
  function highlighted(text, concepts) {
    const spans = [];
    concepts.flat().forEach((term) => {
      for (const match of text.matchAll(phrasePattern(term)))
        spans.push([match.index, match.index + match[0].length]);
    });
    spans.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
    const merged = [];
    for (const span of spans) {
      const last = merged[merged.length - 1];
      if (last && span[0] <= last[1]) last[1] = Math.max(last[1], span[1]);
      else merged.push(span.slice());
    }
    let last = 0,
      result = "";
    for (const [start, end] of merged) {
      result +=
        escape(text.slice(last, start)) +
        `<mark>${escape(text.slice(start, end))}</mark>`;
      last = end;
    }
    return result + escape(text.slice(last));
  }
  function selection(item) {
    const b = baselineById.get(item.item_id);
    const example =
      state.example &&
      audit.score_validity_examples.find(
        (e) => e.question_id === item.item_id && e.excerpt_id === state.example,
      );
    if (example)
      return {
        excerpt: excerpts.get(example.excerpt_id),
        diagnostics: example.diagnostics,
        bare: false,
        label: "Counterexample · passing, unrelated excerpt",
        notice: `${example.claim} ${example.audit_check}`,
        example,
      };
    if (state.mode === "bm25")
      return {
        excerpt: excerpts.get(b.matched_item_id),
        diagnostics: b.diagnostics,
        bare: false,
        label: `BM25 top result · ${b.self_match ? "attached source selected" : "different source selected"}`,
        notice: "",
      };
    if (state.mode === "own")
      return {
        excerpt: excerpts.get(item.item_id),
        diagnostics: item.own_wrapped,
        bare: false,
        label: "The source attached to this question",
        notice: isMetadata(item)
          ? "This attached source passes with its metadata wrapper and fails as a bare excerpt. Use Compare to place this exact excerpt with and without metadata side by side. Metadata dependence alone does not establish a false positive."
          : "",
      };
    const bare = state.mode === "bare";
    const passing = bare
      ? item.passing_bare_excerpt_ids
      : item.passing_excerpt_ids;
    return {
      excerpt: excerpts.get(item.item_id),
      diagnostics: bare ? item.own_bare : item.own_wrapped,
      bare,
      label: `${passing.length} of ${audit.answer_space.n_indexed_excerpts} excerpts can pass · attached source shown`,
      notice: passing.length
        ? `The exhaustive search finds ${passing.length} passing ${bare ? "bare" : "wrapped"} excerpt${passing.length === 1 ? "" : "s"}: ${passing.join(", ")}. The attached source is one passing representative. This is answer-key-aware selection, not a ranker.`
        : `All ${audit.answer_space.n_indexed_excerpts} ${bare ? "bare" : "wrapped"} excerpts were tested; none passes. The attached source is shown as a representative failure. “Best” is the exhaustive pass ceiling, not a claim that this text has the highest partial score.`,
    };
  }
  function renderInspector() {
    const item = byId.get(state.selected),
      b = baselineById.get(item.item_id),
      selected = selection(item),
      d = selected.diagnostics,
      source = selected.excerpt;
    $("question-meta").innerHTML =
      `<span class="item-id">${escape(item.item_id)}</span><span>${escape(human(item.domain))}</span><span>${escape(item.source_clause)}</span>`;
    $("selected-question").textContent = item.question;
    let diagnosis;
    if (!item.passing_excerpt_ids.length)
      diagnosis = `<strong>No indexed excerpt passes this question.</strong> ${b.self_match ? "BM25 already selects the attached source." : "Even replacing the BM25 result cannot produce a pass from this index."} Its own excerpt covers ${pct(item.own_wrapped.concept_coverage)} of the concept groups; the fixed threshold is ${pct(audit.scoring_config.coverage_threshold)}.`;
    else if (!b.passes)
      diagnosis =
        "<strong>A selection miss with a reachable pass.</strong> BM25’s chosen text fails, while at least one indexed excerpt passes the same scoring rule. This is measured headroom for ranking.";
    else
      diagnosis =
        "<strong>BM25 reaches a lexical pass.</strong> Inspect the returned text and the required concepts to judge what that score does—and does not—establish.";
    if (isMetadata(item))
      diagnosis +=
        " <strong>The pass ceiling changes when metadata is removed.</strong>";
    $("question-diagnosis").innerHTML = diagnosis;
    document.querySelectorAll("[data-mode]").forEach((tab) => {
      const active = tab.dataset.mode === state.mode;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    $("evidence-panel").hidden = state.mode === "compare";
    $("comparison-panel").hidden = state.mode !== "compare";
    if (state.mode !== "compare")
      $("evidence-panel").setAttribute("aria-labelledby", `tab-${state.mode}`);
    $("passage-label").textContent = selected.label;
    $("passage-outcome").textContent = d.passes
      ? "LEXICAL PASS"
      : "NO LEXICAL PASS";
    $("passage-outcome").className = `outcome ${d.passes ? "pass" : "fail"}`;
    $("passage-notice").hidden = !selected.notice;
    $("passage-notice").textContent = selected.notice;
    $("source-id").textContent = `EXCERPT ${source.item_id}`;
    const visibleText = selected.bare
      ? source.anchor_text
      : source.wrapped_text;
    // Numeric highlights are display aids, not a second implementation of scoring.
    const highlightGroups = [
      ...item.required_concepts,
      ...(item.numeric_facts || []).map((fact) => [String(fact.value)]),
    ];
    const { prefix, suffix } = sourceParts(source);
    $("source-metadata").hidden = selected.bare;
    $("source-suffix").hidden = selected.bare;
    $("source-metadata").innerHTML = highlighted(prefix, highlightGroups);
    $("source-suffix").innerHTML = highlighted(suffix, highlightGroups);
    $("source-excerpt").innerHTML = highlighted(
      source.anchor_text,
      highlightGroups,
    );
    $("source-link").href = source.source_url;
    $("source-kind").textContent = selected.bare
      ? "BARE EXCERPT"
      : "EXCERPT + WRAPPER";
    const numericPass = d.numeric_coverage === null || d.numeric_coverage === 1;
    $("score-gates").innerHTML =
      `<div class="gate ${d.concept_coverage >= audit.scoring_config.coverage_threshold ? "pass" : "fail"}">Concept coverage<strong>${item.required_concepts.length ? pct(d.concept_coverage) : "No groups"}</strong><small>${item.required_concepts.length ? `Need ≥ ${pct(audit.scoring_config.coverage_threshold)}` : "Gate satisfied by definition"}</small></div><div class="gate ${numericPass ? "pass" : "fail"}">Numeric facts<strong>${d.numeric_coverage === null ? "None required" : pct(d.numeric_coverage)}</strong><small>${d.numeric_coverage === null ? "No numeric gate for this item" : "Every required value must appear"}</small></div><div class="gate ${d.forbidden_asserted.length ? "fail" : "pass"}">Forbidden assertions<strong>${d.forbidden_asserted.length} detected</strong><small>Need zero asserted matches</small></div>`;
    const missing = new Set(
      d.missing_concepts.map((group) => JSON.stringify(group)),
    );
    $("concept-count").textContent =
      `${item.required_concepts.length - d.missing_concepts.length} / ${item.required_concepts.length} groups matched`;
    $("concept-groups").innerHTML = item.required_concepts.length
      ? item.required_concepts
          .map((group) => {
            const absent = missing.has(JSON.stringify(group));
            const onlyWrapper =
              !absent &&
              !selected.bare &&
              !hasLiteral(source.anchor_text, group) &&
              hasLiteral(`${prefix} ${suffix}`, group);
            return `<div class="concept-group ${absent ? "missing" : ""}"><span class="concept-state" aria-label="${absent ? "Missing group" : "Matched group"}">${absent ? "−" : "+"}</span><span>${group.map((term) => `<span class="term ${hasLiteral(visibleText, [term]) ? "matched" : ""}">${escape(term)}</span>`).join(' <span class="or">OR</span> ')}${onlyWrapper ? '<span class="group-location">MATCH IN WRAPPER</span>' : ""}</span></div>`;
          })
          .join("")
      : "<p>No concept groups are required. This key checks numbers and forbidden concepts only.</p>";
    const nums = item.numeric_facts || [],
      forbidden = item.forbidden_concepts || [];
    $("other-rules").innerHTML =
      `<p><strong>Required numeric facts:</strong> ${nums.length ? nums.map((f) => `${escape(f.label)}: ${escape(f.value)} ${escape(f.unit)} (tolerance ${escape(f.tolerance_abs)})`).join("; ") : "None."}</p><p><strong>Missing numeric facts:</strong> ${d.missing_numeric.length ? escape(d.missing_numeric.join("; ")) : "None."}</p><p><strong>Forbidden alternatives:</strong> ${forbidden.length ? forbidden.map((g) => g.map(escape).join(" OR ")).join("; ") : "None."}</p><p><strong>Asserted forbidden matches:</strong> ${d.forbidden_asserted.length ? escape(d.forbidden_asserted.join("; ")) : "None."}</p><p>Numeric matching checks a value’s presence, not its unit or referent. Concept highlights show literal phrase matches; all gate results come from the Python scoring artifact. Outcome code: <code>${escape(d.outcome)}</code>.</p>`;
    $("reference-answer").textContent = item.reference_answer;
    $("selection-provenance").textContent = selected.example
      ? "Verified numeric-referent counterexample · no generated answer"
      : `Deterministic ${selected.bare ? "bare-excerpt" : "wrapped-excerpt"} evaluation · no generated answer`;
    $("export-question").innerHTML =
      `${state.mode === "compare" ? "Export comparison" : "Export this evidence"} <span aria-hidden="true">↓</span>`;
    if (state.mode === "compare") renderComparison(item);
  }
  function sourceParts(source) {
    // The anchor may also appear in the section title, as it does for prd-006.
    // Split on the adapter template boundary instead of the first anchor match.
    const marker = ") states: ";
    const boundary = source.wrapped_text.indexOf(marker);
    const start = boundary + marker.length;
    if (
      boundary < 0 ||
      source.wrapped_text.slice(start, start + source.anchor_text.length) !==
        source.anchor_text
    )
      throw new Error(
        "The indexed wrapper does not match its declared excerpt.",
      );
    return {
      prefix: source.wrapped_text.slice(0, start).trim(),
      suffix: source.wrapped_text
        .slice(start + source.anchor_text.length)
        .trim(),
    };
  }
  function comparisonResponses(item) {
    const own = {
      label: "Own source + metadata",
      excerpt: excerpts.get(item.item_id),
      diagnostics: item.own_wrapped,
      bare: false,
    };
    if (state.comparePair === "metadata")
      return [
        own,
        {
          label: "Same own source, bare",
          excerpt: excerpts.get(item.item_id),
          diagnostics: item.own_bare,
          bare: true,
        },
      ];
    const b = baselineById.get(item.item_id);
    return [
      {
        label: "BM25 retrieved response",
        excerpt: excerpts.get(b.matched_item_id),
        diagnostics: b.diagnostics,
        bare: false,
      },
      own,
    ];
  }
  function renderComparison(item) {
    const responses = comparisonResponses(item),
      [left, right] = responses;
    $("comparison-pair").value = state.comparePair;
    const result = (r) =>
      r.diagnostics.passes ? "lexical pass" : "no lexical pass";
    if (state.comparePair === "metadata") {
      $("comparison-reading").textContent =
        `The same attached-source excerpt (${item.item_id}) and scoring rule; only the metadata wrapper changes. With metadata: ${result(left)}, ${pct(left.diagnostics.concept_coverage)} concept coverage. Bare: ${result(right)}, ${pct(right.diagnostics.concept_coverage)}. A format-sensitive score is not, by itself, a semantic correctness judgment.`;
    } else {
      $("comparison-reading").textContent =
        left.excerpt.item_id === right.excerpt.item_id
          ? "BM25 selects the attached source for this question. These scored responses are identical; changing the selected source is not the issue illustrated by this pair."
          : `Same question, wrapper, and scoring rule; two different selected excerpts. BM25: ${result(left)}. Attached source: ${result(right)}. Compare the original text and missing concept groups below.`;
    }
    const highlights = [
      ...item.required_concepts,
      ...(item.numeric_facts || []).map((f) => [String(f.value)]),
    ];
    $("comparison-responses").innerHTML = responses
      .map((r, index) => {
        const d = r.diagnostics,
          e = r.excerpt;
        const { prefix, suffix } = sourceParts(e);
        const n = item.required_concepts.length,
          missing = d.missing_concepts;
        return `<section class="compare-response" aria-labelledby="compare-label-${index}"><div class="compare-response-heading"><span class="compare-question-id">QUESTION ${escape(item.item_id)}</span><h4 id="compare-label-${index}">${escape(r.label)}</h4><span class="outcome ${d.passes ? "pass" : "fail"}">${d.passes ? "LEXICAL PASS" : "NO LEXICAL PASS"}</span></div><div class="compare-source"><div class="source-sheet-header"><span>EXACT RETURNED TEXT</span><span>EXCERPT ${escape(e.item_id)}</span></div>${r.bare ? '<p class="bare-format-label">No metadata wrapper</p>' : `<p class="source-metadata">${highlighted(prefix, highlights)}</p>`}<blockquote>${highlighted(e.anchor_text, highlights)}</blockquote>${r.bare ? "" : `<p class="source-metadata source-suffix">${highlighted(suffix, highlights)}</p>`}<a class="compare-source-link" href="${escape(e.source_url)}" target="_blank" rel="noopener noreferrer">Source standard <span class="external-arrow" aria-hidden="true"></span></a></div><dl class="compare-gates"><div><dt>Concept coverage</dt><dd class="${d.concept_coverage >= audit.scoring_config.coverage_threshold ? "pass" : "fail"}">${n ? `${pct(d.concept_coverage)} · ${n - missing.length}/${n} groups` : "No groups required"}<small>Need ≥ ${pct(audit.scoring_config.coverage_threshold)}</small></dd></div><div><dt>Numeric coverage</dt><dd class="${d.numeric_coverage === null || d.numeric_coverage === 1 ? "pass" : "fail"}">${d.numeric_coverage === null ? "No facts required" : pct(d.numeric_coverage)}<small>${d.missing_numeric.length ? `Missing: ${escape(d.missing_numeric.join("; "))}` : "No missing numeric facts"}</small></dd></div><div><dt>Forbidden assertions</dt><dd class="${d.forbidden_asserted.length ? "fail" : "pass"}">${d.forbidden_asserted.length} detected<small>${d.forbidden_asserted.length ? escape(d.forbidden_asserted.join("; ")) : "Need zero asserted matches"}</small></dd></div></dl><div class="compare-missing"><h5>Missing concept groups · ${missing.length}</h5>${missing.length ? `<ul>${missing.map((g) => `<li>${g.map(escape).join(' <span class="or">OR</span> ')}</li>`).join("")}</ul>` : "<p>Every required concept group is matched.</p>"}</div><p class="compare-code">Scorer outcome: ${escape(d.outcome)}</p></section>`;
      })
      .join("");
    $("comparison-rubric").innerHTML =
      `<p>Both responses use the same question’s fixed answer key. One alternative satisfies a concept group; at least ${pct(audit.scoring_config.coverage_threshold)} of groups must match. All numeric facts must appear; no forbidden concept may be asserted.</p>${item.required_concepts.length ? `<ol>${item.required_concepts.map((g) => `<li>${g.map(escape).join(" OR ")}</li>`).join("")}</ol>` : "<p>No concept groups are required.</p>"}<p><strong>Numeric facts:</strong> ${(item.numeric_facts || []).map((f) => `${escape(f.label)} = ${escape(f.value)} ${escape(f.unit)} (tolerance ${escape(f.tolerance_abs)})`).join("; ") || "None."}</p><p><strong>Forbidden alternatives:</strong> ${(item.forbidden_concepts || []).map((g) => g.map(escape).join(" OR ")).join("; ") || "None."}</p>`;
    $("comparison-reference").textContent = item.reference_answer;
    $("selection-provenance").textContent =
      "Two exact scored responses · same question and rubric · no generated answers";
  }
  function download(payload, filename) {
    const blob = new Blob([JSON.stringify(payload, null, 2) + "\n"], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast(`Downloaded ${filename}`);
  }
  function focusEvidence() {
    $("inspector").focus({ preventScroll: true });
    $("inspector").scrollIntoView({ block: "start" });
  }
  function wireEvents() {
    $("skip-to-evidence").addEventListener("click", focusEvidence);
    document
      .querySelectorAll("[data-filter]")
      .forEach((button) =>
        button.addEventListener("click", () =>
          setFilter(button.dataset.filter),
        ),
      );
    document
      .querySelectorAll("[data-filter-jump]")
      .forEach((button) =>
        button.addEventListener("click", () =>
          setFilter(button.dataset.filterJump, true),
        ),
      );
    document.querySelectorAll("[data-example]").forEach((button) =>
      button.addEventListener("click", () =>
        selectQuestion(button.dataset.example, {
          mode: "own",
          resetFilter: true,
          scroll: true,
        }),
      ),
    );
    $("question-rows").addEventListener("click", (event) => {
      const button = event.target.closest("[data-question]");
      if (button) {
        selectQuestion(button.dataset.question);
        if (event.detail === 0) focusEvidence();
      }
    });
    $("question-search").addEventListener("input", (event) => {
      state.query = event.target.value;
      const rows = renderLedger();
      if (
        rows.length &&
        !rows.some((item) => item.item_id === state.selected)
      ) {
        state.selected = rows[0].item_id;
        state.example = null;
        renderLedger();
        renderInspector();
      }
      writeUrl();
      announce(`${rows.length} matching questions.`);
    });
    $("reset-view").addEventListener("click", () => {
      state.query = "";
      $("question-search").value = "";
      state.mode = "own";
      state.selected = "loto-001";
      setFilter("all");
    });
    document.querySelectorAll("[data-mode]").forEach((tab, index, tabs) => {
      tab.addEventListener("click", () => {
        const leavingCounterexample =
          Boolean(state.example) && tab.dataset.mode === "compare";
        state.mode = tab.dataset.mode;
        state.example = null;
        renderInspector();
        writeUrl();
        if (leavingCounterexample)
          toast(
            "Counterexample view closed. Comparing the supported source pair.",
          );
        announce(
          state.mode === "compare"
            ? `Comparison selected. ${$("comparison-reading").textContent}`
            : `${tab.textContent.trim()} selected. ${$("passage-outcome").textContent}.`,
        );
      });
      tab.addEventListener("keydown", (event) => {
        let next;
        if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
        if (event.key === "ArrowLeft")
          next = (index + tabs.length - 1) % tabs.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = tabs.length - 1;
        if (next !== undefined) {
          event.preventDefault();
          tabs[next].focus();
          tabs[next].click();
        }
      });
    });
    $("comparison-pair").addEventListener("change", (event) => {
      state.comparePair = event.target.value;
      renderComparison(byId.get(state.selected));
      writeUrl();
      announce($("comparison-reading").textContent);
    });
    $("open-counterexample").addEventListener("click", () =>
      selectQuestion("hc-005", {
        mode: "best",
        example: "hn-001",
        resetFilter: true,
        scroll: true,
      }),
    );
    $("copy-link").addEventListener("click", async () => {
      writeUrl();
      try {
        await navigator.clipboard.writeText(location.href);
        toast("Question link copied.");
      } catch {
        toast("Copy the address bar URL to share this question.");
      }
    });
    $("export-question").addEventListener("click", () => {
      const item = byId.get(state.selected),
        s = selection(item);
      download(
        {
          schema:
            state.mode === "compare"
              ? "evidence_audit_comparison_v1"
              : "evidence_audit_selection_v1",
          evidence_type: audit.evidence_type,
          scope: audit.scope,
          interpretation:
            "A lexical pass is not proof of semantic correctness. Best columns are answer-key-aware diagnostic ceilings.",
          view: state.mode,
          question: item,
          ...(state.mode === "compare"
            ? {
                comparison_pair: state.comparePair,
                responses: comparisonResponses(item).map((r) => ({
                  label: r.label,
                  selected_excerpt: r.excerpt,
                  exact_scored_text: r.bare
                    ? r.excerpt.anchor_text
                    : r.excerpt.wrapped_text,
                  diagnostics: r.diagnostics,
                })),
              }
            : {
                selected_excerpt: s.excerpt,
                exact_scored_text: s.bare
                  ? s.excerpt.anchor_text
                  : s.excerpt.wrapped_text,
                diagnostics: s.diagnostics,
                counterexample: s.example || null,
              }),
          bm25: baselineById.get(item.item_id),
          scoring_config: audit.scoring_config,
          corpus_digest: audit.corpus_digest,
          input_sha256: audit.input_sha256,
          limitations: audit.limitations,
        },
        `evidence-audit-${item.item_id}-${state.mode}.json`,
      );
    });
    $("export-audit").addEventListener("click", () =>
      download(audit, "excerpt-answerability-audit.json"),
    );
    window.addEventListener("hashchange", () => {
      if (location.hash.includes("question=")) {
        readUrl();
        renderLedger();
        renderInspector();
      }
    });
  }
  async function init() {
    try {
      const response = await fetch("evidence-audit-data.json");
      if (!response.ok)
        throw new Error(`Audit artifact returned HTTP ${response.status}`);
      audit = await response.json();
      items = audit.answer_space.items;
      if (audit.schema !== "excerpt_answerability_v1" || !Array.isArray(items))
        throw new Error("Unsupported audit artifact");
      byId = new Map(items.map((i) => [i.item_id, i]));
      excerpts = new Map(audit.excerpts.map((e) => [e.item_id, e]));
      baseline = audit.baselines.find((b) => b.adapter === "retrieval_bm25");
      baselineById = new Map(baseline.items.map((i) => [i.item_id, i]));
      const space = audit.answer_space;
      $("ceiling-count").textContent = space.n_attainable_wrapped;
      $("combination-count").textContent =
        space.n_combinations_per_format.toLocaleString("en-US");
      $("baseline-count").textContent = baseline.n_pass;
      $("recoverable-count").textContent = baseline.n_recoverable_by_reranking;
      $("unreachable-count").textContent = baseline.n_no_excerpt_passes;
      for (const [name, value] of [
        ["baseline", baseline.n_pass],
        ["recoverable", baseline.n_recoverable_by_reranking],
        ["unreachable", baseline.n_no_excerpt_passes],
      ])
        document
          .querySelector(`.decomp-part.${name}`)
          .style.setProperty("--share", value);
      $("ceiling-dots").innerHTML = Array.from(
        { length: space.n_factual },
        (_, index) =>
          `<i class="${index < space.n_attainable_wrapped ? "filled" : ""}"></i>`,
      ).join("");
      $("audit-provenance").innerHTML =
        `Artifact: <a href="evidence-audit-data.json" download>excerpt_answerability_v1 JSON</a>.<br>Reproduce: <code>python3 -m grounding_eval.answerability --check</code><br>Corpus SHA-256: <code>${escape(audit.corpus_digest)}</code><br>${Object.keys(audit.input_sha256).length} input file hashes included in the export.`;
      $("method-experiment").textContent =
        `${space.n_combinations_per_format.toLocaleString("en-US")} combinations per format; ${(space.n_combinations_per_format * 2).toLocaleString("en-US")} evaluations across wrapped and bare formats. ${space.n_attainable_wrapped}/${space.n_factual} questions attainable with the wrapper; ${space.n_attainable_bare}/${space.n_factual} without it.`;
      readUrl();
      renderLedger();
      renderInspector();
      wireEvents();
      if (location.hash.includes("question="))
        $("inspector").scrollIntoView({ block: "start" });
      announce("Audit loaded. 63 factual questions available for inspection.");
      document.documentElement.dataset.auditReady = "true";
    } catch (error) {
      document.documentElement.dataset.auditError = "true";
      for (const id of [
        "ceiling-count",
        "combination-count",
        "baseline-count",
        "recoverable-count",
        "unreachable-count",
      ])
        $(id).textContent = "—";
      $("visible-count").textContent = "Unavailable";
      document.querySelector(".finding-label").textContent =
        "Audit data unavailable. Reload this page or inspect the source repository.";
      document.querySelector(".finding-note").textContent =
        "The interactive figures could not be verified in this view.";
      document.querySelectorAll("button").forEach((button) => {
        button.disabled = true;
      });
      $("question-search").disabled = true;
      $("comparison-pair").disabled = true;
      $("evidence-panel").hidden = true;
      $("comparison-panel").hidden = true;
      $("question-rows").textContent = "";
      $("selected-question").textContent =
        "The audit data could not be loaded.";
      $("question-diagnosis").textContent =
        "Reload this page, or inspect the reproducible analysis in the source repository. No results have been substituted.";
      $("question-diagnosis").classList.add("load-error");
      $("audit-provenance").textContent =
        "Artifact unavailable. See the source repository for the committed data and analysis.";
      announce("The audit artifact could not be loaded.");
      console.error(error);
    }
  }
  init();
})();
