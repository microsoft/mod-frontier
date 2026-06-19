"""Submit ToxicChat T5 inference job to Azure Machine Learning.

Configure the target workspace via environment variables (see constants below).

Usage:
    python submit_aml_job.py [--split test] [--batch-size 32] [--wait]
"""
import argparse
import os

from azure.ai.ml import MLClient, command, Input, Output
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
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    parser.add_argument(
        "--env-name",
        default="toxicchat-t5-env",
        help="AML environment name",
    )
    args = parser.parse_args()

    print("Connecting to Azure ML workspace...")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    print(f"  Workspace: {ml_client.workspace_name}")

    # Create/register the environment
    print("Registering environment...")
    env = Environment(
        name=args.env_name,
        description="Environment for ToxicChat T5-Large inference",
        conda_file=os.path.join(SCRIPT_DIR, "conda.yml"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest",
    )
    env = ml_client.environments.create_or_update(env)
    print(f"  Environment: {env.name}:{env.version}")

    # Build the command
    # The inference script will download model + data from HuggingFace on the compute
    cmd = (
        f"pip install datasets transformers torch scikit-learn sentencepiece protobuf huggingface-hub && "
        f"python download_data.py && "
        f"python inference.py --split {args.split} --batch-size {args.batch_size} "
        f"--output-dir outputs && "
        f"python evaluate.py --predictions outputs/predictions_{args.split}.jsonl "
        f"--output outputs/evaluation_report.json"
    )

    print(f"Submitting job to compute '{COMPUTE_NAME}'...")
    job = command(
        code=SCRIPT_DIR,
        command=cmd,
        environment=f"{env.name}:{env.version}",
        compute=COMPUTE_NAME,
        display_name=f"toxicchat-t5-inference-{args.split}",
        experiment_name="toxicchat-t5-reproduction",
        description=(
            f"Reproduce ToxicChat T5-Large results on {args.split} split. "
            "Paper: arxiv:2310.17389"
        ),
    )

    returned_job = ml_client.jobs.create_or_update(job)
    print(f"\n  Job submitted: {returned_job.name}")
    print(f"  Status: {returned_job.status}")
    print(f"  Studio URL: {returned_job.studio_url}")

    if args.wait:
        print("\nWaiting for job completion...")
        ml_client.jobs.stream(returned_job.name)

        # Download outputs
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
