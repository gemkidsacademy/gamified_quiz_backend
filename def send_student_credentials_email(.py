def send_student_credentials_email(
    to_email: str,
    student_id: str,
):
    print(
        f"[DEBUG] SENDGRID_API_KEY is set: "
        f"{SENDGRID_API_KEY is not None}"
    )
    print(
        f"[DEBUG] SENDGRID_API_KEY length: "
        f"{len(SENDGRID_API_KEY) if SENDGRID_API_KEY else 0}"
    )

    message = Mail(
        from_email="noreply@gemkidsacademy.com.au",
        to_emails=to_email,
        subject="Your Child's Gem Kids Academy Access",
        html_content=f"""
        <div style="
            max-width:650px;
            margin:0 auto;
            padding:20px;
            font-family:Arial, Helvetica, sans-serif;
            color:#333333;
            line-height:1.6;
        ">

            <!-- Logo -->
            <div style="
                text-align:center;
                margin-bottom:30px;
            ">
                <img
                    src="https://gemkidsacademy.com.au/wp-content/uploads/2024/10/cropped-logo-4-1.png"
                    alt="Gem Kids Academy"
                    style="width:180px;"
                />
            </div>

            <!-- Greeting -->
            <p style="font-size:16px;">
                Dear Parent,
            </p>

            <p style="font-size:16px;">
                We're excited to let you know that your child's access to the
                <strong>Gem Kids Digital Learning Platform</strong>
                has now been activated.
            </p>

            <!-- Gem AI Chatbot -->
            <div style="
                margin-top:30px;
                margin-bottom:25px;
            ">

                <h2 style="
                    color:#2c3e50;
                    font-size:21px;
                    margin-bottom:10px;
                ">
                    🤖 Gem AI Chatbot (24/7 AI Learning Assistant)
                </h2>

                <p style="font-size:16px;">
                    Your child can now access <strong>Gem AI</strong>, our
                    AI-powered learning assistant that provides instant
                    academic support anytime, anywhere.
                </p>

                <h3 style="
                    color:#2c3e50;
                    font-size:17px;
                    margin-top:20px;
                ">
                    Login Details:
                </h3>

                <div style="
                    background:#f7f7f7;
                    border:1px solid #e5e5e5;
                    border-radius:8px;
                    padding:18px;
                ">

                    <p style="margin:8px 0;">
                        <strong>Website:</strong>
                        <a
                            href="https://chatbot.gemkidsacademy.com.au"
                            style="color:#008cc8;"
                        >
                            https://chatbot.gemkidsacademy.com.au
                        </a>
                    </p>

                    <p style="margin:8px 0;">
                        <strong>Username:</strong>
                        Your registered parent email address
                    </p>

                    <p style="margin:8px 0;">
                        <strong>Password:</strong>
                        A One-Time Password (OTP) will be sent to your email
                        each time you log in for secure access.
                    </p>

                </div>

            </div>

            <!-- Gamified Quiz Portal -->
            <div style="
                margin-top:35px;
                margin-bottom:25px;
            ">

                <h2 style="
                    color:#2c3e50;
                    font-size:21px;
                    margin-bottom:10px;
                ">
                    🎮 Gem AI Gamified Quiz Portal
                </h2>

                <p style="font-size:16px;">
                    After every class, your child will receive engaging quizzes
                    designed to reinforce learning through fun challenges,
                    points, badges, and leaderboards.
                </p>

                <h3 style="
                    color:#2c3e50;
                    font-size:17px;
                    margin-top:20px;
                ">
                    Login Details:
                </h3>

                <div style="
                    background:#f7f7f7;
                    border:1px solid #e5e5e5;
                    border-radius:8px;
                    padding:18px;
                ">

                    <p style="margin:8px 0;">
                        <strong>Website:</strong>
                        <a
                            href="https://gamifiedquiz.gemkidsacademy.com.au"
                            style="color:#008cc8;"
                        >
                            https://gamifiedquiz.gemkidsacademy.com.au
                        </a>
                    </p>

                    <p style="margin:8px 0;">
                        <strong>Student ID:</strong>
                        {student_id}
                    </p>

                    <p style="margin:8px 0;">
                        <strong>Password:</strong>
                        {student_id}
                    </p>

                </div>

            </div>

            <!-- Features -->
            <div style="
                margin-top:35px;
            ">

                <h3 style="
                    color:#2c3e50;
                    font-size:18px;
                ">
                    What your child can enjoy:
                </h3>

                <p style="font-size:16px; margin:8px 0;">
                    ✅ AI-powered homework assistance
                </p>

                <p style="font-size:16px; margin:8px 0;">
                    ✅ Practice questions aligned with class topics
                </p>

                <p style="font-size:16px; margin:8px 0;">
                    ✅ Interactive revision quizzes
                </p>

                <p style="font-size:16px; margin:8px 0;">
                    ✅ Leaderboards and rewards
                </p>

                <p style="font-size:16px; margin:8px 0;">
                    ✅ Learn anytime, anywhere
                </p>

            </div>

            <!-- Support -->
            <p style="
                font-size:16px;
                margin-top:30px;
            ">
                If you experience any login issues or require assistance,
                please contact our team and we'll be happy to help.
            </p>

            <p style="font-size:16px;">
                Thank you for choosing <strong>Gem Kids Academy</strong>.
            </p>

            <!-- Signature -->
            <p style="
                font-size:16px;
                margin-top:30px;
            ">
                Kind regards,<br><br>

                <strong>Gem Kids Academy</strong>
            </p>

            <hr style="
                margin:35px 0 20px 0;
                border:none;
                border-top:1px solid #e5e5e5;
            ">

            <p style="
                text-align:center;
                font-size:12px;
                color:#777777;
            ">
                © Gem Kids Academy
            </p>

        </div>
        """,
    )

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)

        response = sg.send(message)

        print(
            f"[INFO] Student access email sent to "
            f"{to_email}, status code {response.status_code}"
        )

    except Exception as e:
        print(
            f"[ERROR] Failed to send student access email "
            f"to {to_email}: {e}"
        )
        raise    
