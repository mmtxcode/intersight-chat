import {
  createHash,
  createPrivateKey,
  createSign,
  KeyObject,
} from "node:crypto";

export interface SignedRequestHeaders {
  Host: string;
  Date: string;
  Digest: string;
  "Content-Type": string;
  Authorization: string;
  Accept: string;
}

export interface SigningInput {
  method: string;
  host: string;
  path: string; // request-target path including query string
  body: string; // raw body string ("" for GET)
  contentType?: string;
}

export class IntersightSigner {
  private readonly keyId: string;
  private readonly keyObject: KeyObject;
  private readonly algorithmLabel: string;
  private readonly signNodeAlg: string;

  constructor(keyId: string, pem: string) {
    if (!keyId || !keyId.trim()) {
      throw new Error("Intersight Key ID is required.");
    }
    if (!pem || !pem.includes("-----BEGIN")) {
      throw new Error("Intersight PEM key is required and must be a valid PEM.");
    }
    this.keyId = keyId.trim();
    this.keyObject = createPrivateKey({ key: pem, format: "pem" });

    const keyType = this.keyObject.asymmetricKeyType;
    if (keyType === "ec") {
      // Intersight v3 keys (ECDSA P-256). HS2019 is the modern label per
      // draft-cavage / RFC 9421; Intersight accepts it for EC keys.
      this.algorithmLabel = "hs2019";
      this.signNodeAlg = "sha256";
    } else if (keyType === "rsa") {
      // Intersight v2 keys (RSA).
      this.algorithmLabel = "rsa-sha256";
      this.signNodeAlg = "RSA-SHA256";
    } else {
      throw new Error(
        `Unsupported PEM key type: ${keyType ?? "unknown"} (expected ec or rsa).`,
      );
    }
  }

  sign(input: SigningInput): SignedRequestHeaders {
    const date = new Date().toUTCString();
    const digest =
      "SHA-256=" + createHash("sha256").update(input.body).digest("base64");
    const contentType = input.contentType ?? "application/json";

    const requestTarget = `${input.method.toLowerCase()} ${input.path}`;
    const signingHeaders = [
      "(request-target)",
      "host",
      "date",
      "digest",
      "content-type",
    ];
    const signingString = [
      `(request-target): ${requestTarget}`,
      `host: ${input.host}`,
      `date: ${date}`,
      `digest: ${digest}`,
      `content-type: ${contentType}`,
    ].join("\n");

    const signer = createSign(this.signNodeAlg);
    signer.update(signingString);
    signer.end();
    const signature = signer.sign(this.keyObject).toString("base64");

    const authorization =
      `Signature keyId="${this.keyId}",` +
      `algorithm="${this.algorithmLabel}",` +
      `headers="${signingHeaders.join(" ")}",` +
      `signature="${signature}"`;

    return {
      Host: input.host,
      Date: date,
      Digest: digest,
      "Content-Type": contentType,
      Authorization: authorization,
      Accept: "application/json",
    };
  }
}
