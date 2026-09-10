# Published runs

A run file belongs here only if it records, alongside the scores, the exact
model id, the UTC date it was run, the digest (hash) of the prompts it used,
and which arm (`ungrounded` or `grounded`) produced it. A run file that
cannot be traced back to those four facts is not usable evidence and should
not be committed. Every file here must be a real system run: it must not
carry the `mock__` filename prefix or the mock demonstration banner
(`MOCK_DEMONSTRATION_FIXTURE_NOT_RESULTS`) that fixture runs carry.
