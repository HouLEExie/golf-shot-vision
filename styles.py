def get_css() -> str:
    return """
    <style>
    :root {
        color-scheme: dark;
    }

    .stApp {
        background:
            linear-gradient(135deg, rgba(4, 8, 14, 0.98), rgba(13, 18, 27, 0.96)),
            repeating-linear-gradient(
                90deg,
                rgba(255, 255, 255, 0.035) 0,
                rgba(255, 255, 255, 0.035) 1px,
                transparent 1px,
                transparent 64px
            );
        color: #edf6ff;
    }

    [data-testid="stSidebar"] {
        background: rgba(7, 11, 18, 0.88);
        border-right: 1px solid rgba(125, 214, 255, 0.16);
    }

    [data-testid="stSidebar"] h2 {
        color: #f8fbff;
        font-size: 1rem;
        letter-spacing: 0;
        margin-top: 0.4rem;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1240px;
    }

    .hero-panel {
        position: relative;
        overflow: hidden;
        padding: 2.4rem 2.2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(116, 225, 255, 0.22);
        border-radius: 8px;
        background:
            linear-gradient(120deg, rgba(9, 18, 28, 0.86), rgba(12, 28, 28, 0.72)),
            linear-gradient(90deg, rgba(0, 213, 255, 0.12), rgba(57, 255, 136, 0.08), rgba(255, 210, 63, 0.09));
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.32), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(18px);
    }

    .hero-panel::after {
        content: "";
        position: absolute;
        inset: auto 0 0 0;
        height: 3px;
        background: linear-gradient(90deg, #00d5ff, #39ff88, #ffd23f, #ff3b30);
        opacity: 0.9;
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        min-height: 28px;
        padding: 0 0.7rem;
        border: 1px solid rgba(0, 213, 255, 0.32);
        color: #8eeaff;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
        background: rgba(0, 213, 255, 0.08);
        border-radius: 999px;
    }

    .hero-panel h1 {
        margin: 1rem 0 0.2rem;
        color: #ffffff;
        font-size: clamp(2.4rem, 5vw, 4.8rem);
        line-height: 1;
        letter-spacing: 0;
    }

    .hero-panel h2 {
        margin: 0 0 0.9rem;
        color: #39ff88;
        font-size: clamp(1rem, 2vw, 1.35rem);
        font-weight: 700;
        letter-spacing: 0;
    }

    .hero-panel p {
        max-width: 720px;
        margin: 0;
        color: rgba(237, 246, 255, 0.84);
        font-size: 1.02rem;
        line-height: 1.6;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        color: rgba(237, 246, 255, 0.72);
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 0;
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff;
        background: rgba(0, 213, 255, 0.12);
        border-color: rgba(0, 213, 255, 0.28);
    }

    .tab-intro {
        margin: 0.7rem 0 1rem;
        padding: 0.85rem 1rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.055);
        color: rgba(237, 246, 255, 0.86);
        backdrop-filter: blur(16px);
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 0.9rem 0 1.4rem;
    }

    .metric-card {
        min-height: 128px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 8px;
        background: linear-gradient(160deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.055));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 18px 34px rgba(0, 0, 0, 0.22);
        backdrop-filter: blur(18px);
    }

    .metric-card span {
        display: block;
        color: rgba(237, 246, 255, 0.68);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
    }

    .metric-card strong {
        display: block;
        margin-top: 0.55rem;
        color: #ffffff;
        font-size: clamp(1.2rem, 2vw, 1.9rem);
        line-height: 1.1;
        letter-spacing: 0;
        word-break: break-word;
    }

    .metric-card small {
        display: block;
        margin-top: 0.55rem;
        color: rgba(142, 234, 255, 0.86);
        font-size: 0.82rem;
        line-height: 1.35;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(0, 213, 255, 0.35);
        border-radius: 8px;
        padding: 0.8rem;
        background: rgba(255, 255, 255, 0.045);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid rgba(0, 213, 255, 0.34);
        background: linear-gradient(90deg, rgba(0, 213, 255, 0.92), rgba(57, 255, 136, 0.74));
        color: #021018;
        font-weight: 800;
        letter-spacing: 0;
        min-height: 2.7rem;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: rgba(255, 210, 63, 0.72);
        color: #021018;
        filter: brightness(1.06);
    }

    [data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.08);
    }

    video,
    img {
        border-radius: 8px;
    }

    @media (max-width: 900px) {
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 560px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero-panel {
            padding: 1.5rem 1.2rem;
        }

        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """
