"""Email utilities for sending OTP verification emails."""

import os
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType

# Build connection config from environment variables.
# Defaults are safe no-ops so the app won't crash on startup if vars are missing.
conf = ConnectionConfig(
    MAIL_USERNAME   = os.getenv("MAIL_USERNAME", ""),
    MAIL_PASSWORD   = os.getenv("MAIL_PASSWORD", ""),
    MAIL_FROM       = os.getenv("MAIL_FROM", "noreply@medqa.app"),
    MAIL_PORT       = int(os.getenv("MAIL_PORT", "587")),
    MAIL_SERVER     = os.getenv("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_STARTTLS   = os.getenv("MAIL_STARTTLS", "true").lower() == "true",
    MAIL_SSL_TLS    = os.getenv("MAIL_SSL_TLS", "false").lower() == "true",
    USE_CREDENTIALS = True,
    VALIDATE_CERTS  = True,
)

_mail = FastMail(conf)


async def send_verification_email(email: str, otp: str) -> None:
    """
    Send a 6-digit OTP verification email.

    Parameters
    ----------
    email : str  – recipient email address
    otp   : str  – 6-digit one-time password
    """
    html_body = f"""
    <div style="font-family: 'Helvetica Neue', Arial, sans-serif; max-width: 480px;
                margin: 0 auto; background: #fffdf9; border: 1px solid #ede8df;
                border-radius: 16px; overflow: hidden;">
      <!-- Header -->
      <div style="background: linear-gradient(135deg, #c8102e 0%, #6b0518 70%, #2d4e38 100%);
                  padding: 32px 36px;">
        <div style="display: flex; align-items: center; gap: 12px;">
          <div style="width: 36px; height: 36px; background: rgba(255,255,255,0.2);
                      border-radius: 10px; display: flex; align-items: center;
                      justify-content: center; font-size: 18px;">❤️</div>
          <span style="color: #fff; font-weight: 700; font-size: 18px; letter-spacing: -0.3px;">
            MedQA Assistant
          </span>
        </div>
      </div>

      <!-- Body -->
      <div style="padding: 36px;">
        <h2 style="margin: 0 0 8px; color: #2e261d; font-size: 22px; font-weight: 700;">
          Verify your email address
        </h2>
        <p style="margin: 0 0 28px; color: #9b8f85; font-size: 14px; line-height: 1.6;">
          Use the code below to verify your account. It expires in <strong>10 minutes</strong>.
        </p>

        <!-- OTP box -->
        <div style="background: #f5f0e8; border: 1.5px dashed #c8a98a; border-radius: 12px;
                    padding: 24px; text-align: center; margin-bottom: 28px;">
          <p style="margin: 0 0 4px; color: #9b8f85; font-size: 11px;
                    font-weight: 600; text-transform: uppercase; letter-spacing: 2px;">
            Your one-time code
          </p>
          <p style="margin: 0; color: #c8102e; font-size: 40px; font-weight: 800;
                    letter-spacing: 12px; font-variant-numeric: tabular-nums;">
            {otp}
          </p>
        </div>

        <p style="margin: 0; color: #c8a98a; font-size: 12px; line-height: 1.6;">
          If you didn't create an account on MedQA Assistant, you can safely ignore this email.
        </p>
      </div>

      <!-- Footer -->
      <div style="padding: 16px 36px; border-top: 1px solid #ede8df; background: #f5f0e8;">
        <p style="margin: 0; color: #c8a98a; font-size: 11px;">
          MedQA is for informational purposes only and is not a substitute for professional medical advice.
        </p>
      </div>
    </div>
    """

    message = MessageSchema(
        subject="Your MedQA verification code",
        recipients=[email],
        body=html_body,
        subtype=MessageType.html,
    )

    await _mail.send_message(message)
