import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def make_animation(
    env,
    trajectories,
    dt,
    dt_last,
    duration=15,
    filename="trajectories.mp4",
    n_vort=200,
    n_vel=36,
    cmap="viridis",
    dpi=150,
):
    """
    Создаёт видео с полем завихренности, полем скорости
    и траекториями частиц.

    Parameters
    ----------
    env : environment
        Окружение с env.vel_field, x_min, x_max, etc.

    trajectories : list of np.ndarray
        Список траекторий.
        Каждая trajectory имеет shape (N, 2).

        Все траектории, кроме последней, имеют шаг dt.
        Последняя траектория имеет шаг dt_last.

    dt : float
        Шаг времени для всех траекторий, кроме последней.

    dt_last : float
        Шаг времени для последней траектории.

    duration : float
        Желаемая продолжительность видео в секундах.

    filename : str
        Имя выходного MP4 файла.

    n_vort : int
        Размер сетки для vorticity.

    n_vel : int
        Размер сетки для velocity field.

    cmap : str
        Colormap для vorticity.

    dpi : int
        Разрешение видео.
    """

    # =========================================================
    # 1. Проверяем FFmpeg
    # =========================================================

    if not FFMpegWriter.isAvailable():
        raise RuntimeError(
            "FFmpeg не найден.\n"
            "Установи его командой:\n\n"
            "conda install -c conda-forge ffmpeg"
        )

    # =========================================================
    # 2. Времена для каждой траектории
    # =========================================================

    trajectory_times = []

    for i, traj in enumerate(trajectories):

        if i == len(trajectories) - 1:
            dt_traj = dt_last
        else:
            dt_traj = dt

        times = np.arange(len(traj)) * dt_traj

        trajectory_times.append(times)

    # Максимальное физическое время
    t_end = max(
        times[-1]
        for times in trajectory_times
        if len(times) > 0
    )

    print(f"Physical time: 0 ... {t_end:.4f} s")

    # =========================================================
    # 3. Количество кадров
    # =========================================================

    # Около 30 кадров/сек для готового видео.
    # Можно увеличить до 60, если нужна более плавная анимация.
    fps = 30

    n_frames = int(np.round(duration * fps))

    # Реальное время между соседними кадрами видео
    dt_video = t_end / (n_frames - 1)

    print(f"Video duration: {duration:.2f} s")
    print(f"FPS: {fps}")
    print(f"Frames: {n_frames}")
    print(f"Video dt: {dt_video:.6f} s")

    # =========================================================
    # 4. Figure
    # =========================================================

    fig, ax = plt.subplots(figsize=(8, 7))

    # =========================================================
    # 5. Vorticity field
    # =========================================================

    x = np.linspace(
        env.x_min,
        env.x_max,
        n_vort
    )

    y = np.linspace(
        env.y_min,
        env.y_max,
        n_vort
    )

    X, Y = np.meshgrid(x, y)

    # Начальное поле
    VORT = np.vectorize(
        env.vel_field.vort
    )(X, Y, 0)

    # ---------------------------------------------------------
    # ВАЖНО:
    # фиксируем цветовую шкалу для всего видео
    # ---------------------------------------------------------

    vmax = np.max(np.abs(VORT))

    # Если поле меняется существенно со временем,
    # можно найти глобальный vmax заранее.
    #
    # Здесь пока используем значение в t=0.
    # Если нужно, ниже покажу более точный вариант.

    im = ax.pcolormesh(
        X,
        Y,
        VORT,
        shading="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        alpha=0.85,
        zorder=1,
        rasterized=True,
    )

    # =========================================================
    # 6. Colorbar
    # =========================================================

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_ticks([-70, -35, 0, 35, 70])
    cbar.ax.tick_params(labelsize=20)

    # =========================================================
    # 7. Velocity field
    # =========================================================

    xq = np.linspace(
        env.x_min,
        env.x_max,
        n_vel
    )

    yq = np.linspace(
        env.y_min,
        env.y_max,
        n_vel
    )

    Xq, Yq = np.meshgrid(xq, yq)

    U = np.vectorize(
        env.vel_field.velX
    )(Xq, Yq, 0)

    V = np.vectorize(
        env.vel_field.velY
    )(Xq, Yq, 0)

    quiver = ax.quiver(
        Xq,
        Yq,
        U,
        V,
        color="k",
        alpha=0.8,
        zorder=2,
        rasterized=True,
    )

    # =========================================================
    # 8. Trajectories
    # =========================================================

    colors = list(
        plt.cm.jet(
            np.linspace(
                0,
                1,
                len(trajectories) - 1
            )
        )
    )

    colors.append("white")

    lines = []
    points = []

    for color in colors:

        line, = ax.plot(
            [],
            [],
            lw=2,
            color=color,
            zorder=3,
        )

        point, = ax.plot(
            [],
            [],
            marker="o",
            markersize=5,
            color=color,
            zorder=3,
        )

        lines.append(line)
        points.append(point)

    # =========================================================
    # 9. Obstacles
    # =========================================================

    ax.add_patch(
        plt.Circle(
            (env.xA, env.yA),
            env.rA,
            color="tomato",
            alpha=1,
            zorder=4,
        )
    )

    ax.add_patch(
        plt.Circle(
            (env.xB, env.yB),
            env.rB,
            color="tomato",
            alpha=1,
            zorder=4,
        )
    )

    ax.text(
        env.xA,
        env.yA,
        "A",
        color="black",
        fontsize=20,
        ha="center",
        va="center",
        zorder=5,
    )

    ax.text(
        env.xB,
        env.yB,
        "B",
        color="black",
        fontsize=20,
        ha="center",
        va="center",
        zorder=5,
    )

    # =========================================================
    # 10. Formatting
    # =========================================================

    ax.set_xlim(
        env.x_min,
        env.x_max
    )

    ax.set_ylim(
        env.y_min,
        env.y_max
    )

    ax.set_aspect("equal")

    ax.set_xlabel(
        "x",
        fontsize=32
    )

    ax.set_ylabel(
        "y",
        fontsize=32
    )

    ax.set_xticks([0, 2, 4, 6])
    ax.set_yticks([0, 2, 4, 6])

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=24,
    )

    ax.grid(False)

    # Время на картинке
    time_text = ax.text(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=20,
        color="black",
        zorder=10,
    )

    plt.tight_layout()

    # =========================================================
    # 11. Update function
    # =========================================================

    def update(frame):

        # -----------------------------------------------------
        # Физическое время текущего кадра
        # -----------------------------------------------------

        time = frame * dt_video

        # Не выходим за t_end из-за численных ошибок
        time = min(time, t_end)

        # -----------------------------------------------------
        # Vorticity
        # -----------------------------------------------------

        VORT = np.vectorize(
            env.vel_field.vort
        )(X, Y, time)

        im.set_array(
            VORT.ravel()
        )

        # -----------------------------------------------------
        # Velocity field
        # -----------------------------------------------------

        U = np.vectorize(
            env.vel_field.velX
        )(Xq, Yq, time)

        V = np.vectorize(
            env.vel_field.velY
        )(Xq, Yq, time)

        quiver.set_UVC(U, V)

        # -----------------------------------------------------
        # Trajectories
        # -----------------------------------------------------

        for (
            traj,
            traj_times,
            line,
            point,
        ) in zip(
            trajectories,
            trajectory_times,
            lines,
            points,
        ):

            # -------------------------------------------------
            # Траектория ещё существует
            # -------------------------------------------------

            if time <= traj_times[-1]:

                # Индекс последней точки,
                # которая уже достигнута
                i = np.searchsorted(
                    traj_times,
                    time,
                    side="right",
                )

                # Не меньше одной точки
                i = max(1, i)

                # Рисуем пройденную часть траектории
                line.set_data(
                    traj[:i, 0],
                    traj[:i, 1],
                )

                # Интерполяция текущего положения
                x_current = np.interp(
                    time,
                    traj_times,
                    traj[:, 0],
                )

                y_current = np.interp(
                    time,
                    traj_times,
                    traj[:, 1],
                )

                point.set_data(
                    [x_current],
                    [y_current],
                )

            # -------------------------------------------------
            # Траектория уже закончилась
            # -------------------------------------------------

            else:

                # Оставляем всю траекторию
                line.set_data(
                    traj[:, 0],
                    traj[:, 1],
                )

                # Частица остаётся в конечной точке
                point.set_data(
                    [traj[-1, 0]],
                    [traj[-1, 1]],
                )

        # -----------------------------------------------------
        # Время
        # -----------------------------------------------------

        time_text.set_text(
            f"t = {time:.3f} s"
        )

        return (
            [im, quiver, time_text]
            + lines
            + points
        )

    # =========================================================
    # 12. Создаём animation
    # =========================================================

    animation = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
    )

    # =========================================================
    # 13. Save
    # =========================================================

    print("Saving video...")

    writer = FFMpegWriter(
        fps=fps,
        metadata={
            "title": "Velocity field and trajectories"
        },
        bitrate=5000,
    )

    animation.save(
        filename,
        writer=writer,
        dpi=dpi,
    )

    plt.close(fig)

    print(f"Saved: {filename}")

    return animation