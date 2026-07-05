"""Deploys contracts/crisis_negotiator.py to a running GenLayer node.

    python -m agent.deploy --chain localnet
    python -m agent.deploy --chain testnet_asimov --private-key 0x...

The GenLayer CLI (`npm install -g genlayer`, then `genlayer deploy
--contract contracts/crisis_negotiator.py`) is the documented, officially
supported deploy path and is what docs.genlayer.com and GenLayer Studio
point you to -- prefer it if you have Node available. This script is a
minimal genlayer-py-only alternative for a pure-Python workflow.

NOTE on reading back the deployed address: deploy_contract() and
write_contract() both return a transaction hash, not an address (confirmed:
genlayer_py/contracts/actions.py, _send_transaction() returns the
consensus contract's NewTransaction event's txId). The deployed contract's
own address is not exposed under one single stable field name across
localnet vs. testnet in genlayer-py 0.8.1's GenLayerTransaction shape, so
this script prints the full simplified receipt and tries a couple of
likely field names as a convenience -- verify against that printed receipt
(or `genlayer deploy`'s own output) rather than trusting the guess blindly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "contracts" / "crisis_negotiator.py"


def main(argv: list[str] | None = None) -> None:
    # Deferred import: genlayer-py requires Python >= 3.12 and has no reason
    # to be installed for MockEnv-only work.
    from genlayer_py import create_account, create_client
    from genlayer_py.chains import localnet, studionet, testnet_asimov, testnet_bradbury
    from genlayer_py.types import TransactionStatus

    chains = {
        "localnet": localnet,
        "testnet_asimov": testnet_asimov,
        "testnet_bradbury": testnet_bradbury,
        "studionet": studionet,
    }

    parser = argparse.ArgumentParser(description="Deploy CrisisNegotiator.")
    parser.add_argument("--chain", default="localnet", choices=sorted(chains))
    parser.add_argument("--private-key", default=os.environ.get("GENLAYER_PRIVATE_KEY"))
    parser.add_argument("--contract-path", default=str(CONTRACT_PATH))
    args = parser.parse_args(argv)

    account = create_account(args.private_key) if args.private_key else create_account()
    client = create_client(chain=chains[args.chain], account=account)

    # studionet shares localnet's chain id 61999, so fund_account works on
    # both (it only refuses when chain.id != localnet.id).
    if args.chain in ("localnet", "studionet"):
        try:
            client.fund_account(address=account.address, amount=10**18)
        except Exception as exc:
            print(f"fund_account skipped: {exc}")

    code = Path(args.contract_path).read_text()
    print(f"Deploying {args.contract_path} to {args.chain} as {account.address} ...")

    tx_hash = client.deploy_contract(code=code, account=account, args=[])
    receipt = client.wait_for_transaction_receipt(
        transaction_hash=tx_hash,
        status=TransactionStatus.ACCEPTED,
        interval=3000,
        retries=40,
    )

    address = (
        receipt.get("recipient")
        or receipt.get("to_address")
        or (receipt.get("tx_data_decoded") or {}).get("contract_address")
    )

    print(json.dumps(receipt, indent=2, default=str))
    if address:
        print(f"\nDeployed contract address (best-effort guess, verify above): {address}")
        print(
            f"Run training against it with:\n  python -m agent.train --env genlayer --address {address}"
        )
    else:
        print(
            "\nCould not find an obvious address field in the receipt above. "
            "Inspect it manually, or use `genlayer deploy` / GenLayer Studio instead."
        )


if __name__ == "__main__":
    main()
