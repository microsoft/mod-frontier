"""Submit T5 inference on a custom JSONL field to Azure ML.

Usage:
    python submit_field_job.py \
        --data ../data/toxicchat_with_relevance.jsonl \
        --field model_output_gpt5 \
        --label T5_model_output_gpt5 \
        --wait
"""
import argparse
import os
import shutil
import tempfile

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# Azure ML workspace configuration. Set these via environment variables, or edit
# the placeholder defaults below to point at your own workspace.
SUBSCRIPTION_ID = os.environ.get("AML_SUBSCRIPTION_ID", "<your-subscription-id>")
RESOURCE_GROUP = os.environ.get("AML_RESOURCE_GROUP", "<your-resource-group>")
WORKSPACE_NAME = os.environ.get("AML_WORKSPACE_NAME", "<your-workspace>")
COMPUTE_NAME = os.environ.get("AML_COMPUTE_NAME", "<your-gpu-compute>")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to input JSONL file")
    parser.add_argument("--field", required=True, help="JSONL field to classify")
    parser.add_argument("--label", required=True, help="Output column name")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--env-name", default="toxicchat-t5-env")
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    data_filename = os.path.basename(data_path)

    print("Connecting to Azure ML workspace...")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    print(f"  Workspace: {ml_client.workspace_name}")

    print("Registering environment...")
    env = Environment(
        name=args.env_name,
        description="Environment for ToxicChat T5-Large inference",
        conda_file=os.path.join(SCRIPT_DIR, "conda.yml"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest",
    )
    env = ml_client.environments.create_or_update(env)
    print(f"  Environment: {env.name}:{env.version}")

    # Bundle data file with code to avoid storage permission issues
    staging_dir = tempfile.mkdtemp(prefix="aml_staging_")
    print(f"  Staging dir: {staging_dir}")
    # Copy all Python scripts
    for f in os.listdir(SCRIPT_DIR):
        src = os.path.join(SCRIPT_DIR, f)
        if os.path.isfile(src):
            shutil.copy2(src, staging_dir)
    # Copy data file into staging dir
    shutil.copy2(data_path, staging_dir)
    print(f"  Copied data file: {data_filename}")

    cmd = (
        f"pip install datasets transformers torch scikit-learn sentencepiece protobuf huggingface-hub && "
        f"python inference_field.py "
        f"--input {data_filename} "
        f"--output outputs/{data_filename} "
        f"--field {args.field} "
        f"--label {args.label} "
        f"--batch-size {args.batch_size}"
    )

    print(f"Submitting job to compute '{COMPUTE_NAME}'...")
    job = command(
        code=staging_dir,
        command=cmd,
        environment=f"{env.name}:{env.version}",
        compute=COMPUTE_NAME,
        display_name=f"t5-inference-{args.field}-{args.label}",
        experiment_name="toxicchat-t5-custom-field",
        description=f"T5 inference on field '{args.field}' -> '{args.label}'",
    )

    returned_job = ml_client.jobs.create_or_update(job)
    print(f"\n  Job submitted: {returned_job.name}")
    print(f"  Status: {returned_job.status}")
    print(f"  Studio URL: {returned_job.studio_url}")

    if args.wait:
        print("\nWaiting for job completion...")
        ml_client.jobs.stream(returned_job.name)

        output_dir = os.path.join(SCRIPT_DIR, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nDownloading outputs to {output_dir}...")
        ml_client.jobs.download(
            returned_job.name,
            download_path=output_dir,
            output_name="default",
        )
        print("Done. Check outputs/ for results.")


if __name__ == "__main__":
    main()
