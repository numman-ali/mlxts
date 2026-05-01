# examples/image-proof

This folder owns host-side image proof artifact verification shared by Phase 10
image workbooks.

The verifier checks reports and BMP files produced by example proof commands.
It does not score visual quality, rerun generation, or define diffusion package
contracts.

Structured verifier output stays AXI-shaped. Usage errors exit `2`, validation
failures exit `1`, and successful verification exits `0`.
