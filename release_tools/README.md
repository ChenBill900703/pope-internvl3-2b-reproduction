# Release checks

`check_public.py` checks the public allowlist, aggregate data, frozen rules, links and SHA manifest. `verify_local.py` additionally needs private original research artifacts and reads them without starting a model. `plot_public.py` redraws figures from public aggregates. `package_public.py` verifies the public-only ZIP after extraction. Package validation does not establish a successful GitHub upload.
