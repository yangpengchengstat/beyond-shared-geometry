# Data and reproducibility scope

Included: published full-precision scalar outcomes, aggregate and per-condition
records, frozen historical result tables, failed/stopped branch records, source
code and configurations, software tests and source/checksum mappings. The
paper/supplementary and review_reproduction capsules support saved-output/table
inspection without model weights or original activation caches.

Not bundled: model weights, raw activation tensors, the multi-gigabyte original
result archives and private local diagnostics. results/registry.json identifies
external archives; study input mappings specify required hashes and members.
No public download location or unrestricted availability is asserted. An author
must supply an actual permitted archive location to reproduce analyses requiring
those inputs. Do not substitute unrelated data or treat hashes as data access.

The historical README is retained under release/history/before_completion_v2/;
its ../../ local-archive examples describe the earlier machine layout and are
not guaranteed paths on a new computer. Supply explicit archive/root arguments
as documented by each study rather than silently downloading or refitting.

Software tests, complete numerical reproduction, and extraction are separate
operations. This packaging run executes none of them. Saved historical reports
are not new test receipts. All previously recorded outcomes remain unchanged.
