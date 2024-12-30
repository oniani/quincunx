//! Multithreaded Quincunx Simulations

#![warn(clippy::all, clippy::pedantic, missing_docs)]

use anyhow::{Context, Result};
use plotters::prelude::*;

/// Represents the quincunx board
#[derive(Debug)]
pub struct Board {
    /// Number of slots in the board
    pub n_slots: usize,
    /// Number of levels in the board
    pub n_levels: usize,
    /// Number of partcles in the board
    pub n_particles: u32,
    /// Initial position for new particles
    pub initial_pos: usize,
    /// Board slots
    pub slots: Vec<u32>,
}

/// Impelments the quincunx board
impl Board {
    #[must_use]
    fn new(n_slots: usize, n_particles: u32) -> Self {
        let mut slots = vec![0; n_slots];
        let n_levels = n_slots / 2;

        let initial_pos = n_slots / 2;
        slots[initial_pos] = n_particles;

        Board {
            n_slots,
            n_levels,
            n_particles,
            initial_pos,
            slots,
        }
    }

    #[must_use]
    fn particles(&self) -> u32 {
        self.slots.iter().sum()
    }
}

/// Defines the direction
#[derive(Clone, Copy, Debug)]
enum Direction {
    Left,
    Right,
    Middle,
}

impl Direction {
    /// `random` performs a random move.
    ///
    /// A random move is made either to the left, to the right, or to the middle (i.e., in-between).
    /// Note that the randomization has weights. This is due to how the probabilities will be
    /// arranged. We have the probability of 1/4 that the particle will end up on the left peg,
    /// 0.25 that it will end up on the right peg, and 1/4 + 1/4 = 0.5 probability that it will end
    /// up not moving horizontally. Notice that even though the particle does not move
    /// horizontally, it obviously does move vertically as it goes down the levels of pegs.
    ///
    /// # Errors
    ///
    /// If the number of elements or false positive probability is not valid.
    ///
    /// # Example
    ///
    /// ```
    /// let random_direction = Direction::random();
    /// ```
    #[must_use]
    fn random() -> Result<Direction> {
        let choices = [Direction::Left, Direction::Right, Direction::Middle];
        let weights = [0.25, 0.25, 0.5];

        let dist = rand::distributions::WeightedIndex::new(&weights)?;
        let mut rng = rand::prelude::thread_rng();
        let direction = choices[rand::prelude::Distribution::sample(&dist, &mut rng)];

        Ok(direction)
    }
}

/// Represents a particle.
pub struct Particle<'a> {
    /// `Board` is wrapped inside a `Mutex` and `Arc`
    ///
    /// - `Mutex` provides synchronization
    /// - `Arc` provides lifetimes, so each thread participates in ownership over the `Mutex<Board>`
    pub board: std::sync::Arc<std::sync::Mutex<Board>>,
    /// Name of the particle
    pub name: &'a str,
    /// Position of the particle
    pub pos: usize,
}

impl<'a> Particle<'a> {
    fn new(board: std::sync::Arc<std::sync::Mutex<Board>>, name: &'a str) -> Result<Self> {
        let pos = board.lock().unwrap().initial_pos;
        Ok(Particle { board, name, pos })
    }

    /// Notice that in quincunx board, there is a case when the particle position is in-between the
    /// cells. This is obviously not convenient for us since in the end, every particle must have
    /// some cell to reside. Because of this, we skip every other level starting with the level 0.
    /// Notice that, in this case, we will always have a third case in which the particle position
    /// has not changed. Look at the level 0 and level 1. It is easy to see that there is a case
    /// where the particle has not changed its location. It seems like we are jumping from level 0
    /// to level 1, but in reality, we are accounting for this by having a third move option which
    /// is "stay in the same position" or "not move". Once again, look at level 0 and level 1 where
    /// the particle does not change its position. This is due to the fact that on the level
    /// in-between level 0 and level 1 the particle could have gone back to its initial position.
    ///
    /// 0                         *
    ///                         *   *
    /// 1                     *   *   *
    ///                     *   *   *   *
    /// 3                 *   *   *   *   *
    ///                 *   *   *   *   *   *
    /// 4             *   *   *   *   *   *   *
    ///             *   *   *   *   *   *   *   *
    /// 5         *   *   *   *   *   *   *   *   *
    ///         *   *   *   *   *   *   *   *   *   *
    ///     |___|___|___|___|___|___|___|___|___|___|___|
    ///
    /// In this case, the particle does not really move neither to the left, nor to the right. We
    /// could update our position by adding or subtracting 0.5, but for convenience and simplicity,
    /// it's the best to simply not change the position of the particle.
    fn move_deterministic(&mut self, direction: Direction) -> Result<()> {
        let mut board = self.board.lock().unwrap();
        board.slots[self.pos] -= 1;
        match direction {
            Direction::Left => {
                if self.pos == 0 {
                    self.pos = 1
                } else {
                    self.pos -= 1;
                }
            }
            Direction::Right => {
                if self.pos == board.n_slots - 1 {
                    self.pos = board.n_slots - 2
                } else {
                    self.pos += 1
                }
            }
            Direction::Middle => {}
        }
        board.slots[self.pos] += 1;
        Ok(())
    }

    fn move_random(&mut self) -> Result<()> {
        self.move_deterministic(Direction::random()?)
    }

    fn simulate(&mut self) -> Result<()> {
        let n_levels = (&self).board.lock().unwrap().n_levels;
        for _ in 0..n_levels {
            self.move_random()?;
        }
        Ok(())
    }
}

/// Creates a bar graph
pub fn plot(data: Vec<i32>, n_particles: u32, path: &std::path::Path) -> Result<()> {
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&full_palette::GREY_200)?;

    let mut chart = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption(
            format!("{:?} Threaded Particles", n_particles),
            ("monospace", 32),
        )
        // 11 as X as we have 11 slots and one more than maximum for y
        .build_cartesian_2d(
            (0..11).into_segmented(),
            0..*data.iter().max().context("Empty iterator")? + 1,
        )?;

    chart.configure_mesh().draw()?;

    chart.draw_series((0..).zip(data.iter()).map(|(x, y)| {
        let x0 = SegmentValue::Exact(x);
        let x1 = SegmentValue::Exact(x + 1);
        let mut bar = Rectangle::new([(x0, 0), (x1, *y)], full_palette::RED_400.filled());
        bar.set_margin(0, 0, 5, 5);
        bar
    }))?;

    Ok(())
}

fn main() -> Result<()> {
    // Parse command line arguments
    let n_slots = std::env::args()
        .nth(1)
        .context("Number of slots required")?
        .parse::<usize>()?;
    let n_particles = std::env::args()
        .nth(2)
        .context("Number of particles required")?
        .parse::<u32>()?;
    let path_bar_graph = std::env::args()
        .nth(3)
        .context("Path to bar graph required")?
        .parse::<String>()?;

    // Initialize the board
    let board = std::sync::Arc::new(std::sync::Mutex::new(Board::new(n_slots, n_particles)));

    // Simulate particles falling down the quincunx board via multithreading
    let handles: Vec<_> = (0..n_particles)
        .map(|idx| {
            let clone = board.clone();
            std::thread::spawn(move || {
                Particle::new(clone, format!("Particle #{}", idx).as_ref())?.simulate()
            })
        })
        .collect();

    for handle in handles {
        let _ = handle.join().unwrap();
    }

    // Obtain the board and ensure that the total number of particles is the same as specified
    let mut board = board.lock().unwrap();
    assert_eq!(n_particles, board.particles());

    // Plot the board as a bar graph
    let path_plot = std::path::Path::new(&path_bar_graph);
    std::fs::create_dir_all(
        path_plot
            .parent()
            .context("Path terminated in a root prefix or is an empty string")?,
    )
    .context("Directory could not be created")?;
    plot(
        board.slots.iter_mut().map(|x| *x as i32).collect(),
        n_particles,
        path_plot,
    )
}
