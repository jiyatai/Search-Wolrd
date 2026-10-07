# Uploading this cleaned repository

The inherited `origin` remote has been removed. Create an empty repository at
your hosting provider, then run from this directory:

```bash
git remote add origin <YOUR_NEW_REPOSITORY_URL>
git add .
git commit -m "Align action space and paper training configuration"
git branch -M main
git push -u origin main
```

Before publishing, replace `REPLACE_WITH_YOUR_REPOSITORY_URL` in `setup.py`,
record the UAV-ON commit used for evaluation, and publish checkpoints/data
separately with hashes. Do not commit simulator environments, credentials,
internal server paths or multi-gigabyte checkpoints.
