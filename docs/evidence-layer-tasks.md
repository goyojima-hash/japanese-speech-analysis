# Common Evidence Layer Tasks

1. Add failing tests for evidence composition, cache behaviour, and Range evidence consumption.
2. Create immutable evidence models and serialization.
3. Extract shared SudachiPy tokenisation while preserving Range imports and output.
4. Add a speech adapter and additive mora timing output to the existing Fluency path.
5. Implement an opt-in provenance-keyed cache and evidence pipeline.
6. Integrate the pipeline into `api_service.py` without changing existing response fields.
7. Run focused tests, then the full suite; review the diff for accidental scoring or persistent evidence writes.
