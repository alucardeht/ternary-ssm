# Hardware And Batch Policy

The old "batch_size > 4 is forbidden" rule was a crash-prevention guard, not a
research truth.

It existed because previous agents jumped directly to large values such as
batch 256 without first measuring small cases, repeatedly crashing or rebooting
the user's PC. The real rule is progressive, measured hardware flex.

## Principle

Use as much available hardware as is safe and useful, but discover the safe
region from the bottom up.

Consumer hardware is the target, not a toy constraint. If a small diagnostic
uses only a few hundred MB, a real training run should be allowed to scale
toward the available 12 GB VRAM when measurements show that it is safe.

Low VRAM use does not imply low user impact. A Vulkan training probe can use
only a few GB of VRAM while still saturating compute at 100% and causing video
playback or desktop stutter. Treat GPU compute saturation as a separate
resource from VRAM.

## Long GPU Run Consent

Before any training, validation, benchmark, or probe that is expected to run for
more than a few minutes, or that may drive the GPU near full compute
utilization, ask the user which mode to use:

- `interactive`: keep the PC usable, prefer the shortest identifying probe,
  leave compute headroom, and stop before repeated long runs;
- `exclusive`: the user is not using the PC, so larger batches, longer warmups,
  or first-refresh tests may run with the normal ramp protocol.

Do not start a long normal-refresh run, long warmup, long benchmark, or
large-batch sweep without this consent. If consent is not available, default to
interactive mode and use only short smoke tests or build/code review.

When the user is using the PC, preserve a small responsiveness margin even if
VRAM appears underutilized. Prefer bounded tests, fewer steps, smaller datasets,
or a paused/deferred run over keeping the GPU at full compute occupancy.

## Exclusive Overnight Validation

When the user explicitly grants exclusive overnight use of the PC, treat the run
as a monitored validation job, not as a fire-and-forget command.

Before launch:

1. check that no project `train.exe`, benchmark, or validation run is active;
2. confirm `nvidia-smi` telemetry works;
3. compile the target executable;
4. run a short smoke of the exact runner/log/analyzer path;
5. verify expected activation lines, eval lines, analyzer parseability, and
   clean process exit;
6. write the job manifest, command list, stop thresholds, and output paths;
7. checkpoint any runner/rule changes before starting the long run.

During launch and execution:

- run arms sequentially, never concurrently;
- monitor GPU utilization, VRAM, temperature, power, process liveness, log
  growth, and exit code;
- write telemetry to files under the workspace `.work/jobs/...` area;
- stop the active arm if temperature, memory, stalled logs, missing process, or
  nonzero exit indicates unsafe or invalid execution;
- after every arm, verify expected completion and activation/eval lines before
  starting the next arm.

After execution:

- run the analyzer from saved logs;
- record the analyzer decision and the exact artifact paths;
- check again that no project `train.exe` remains alive;
- update the relevant research summary/state before reporting completion.

## Process Hygiene

Before starting any training, validation, benchmark, or GPU probe, check whether
another project training process is already running. If any `train.exe`,
benchmark, worker, or long-running validation process from this workspace is
active, do not start a second one until its purpose and ownership are clear.

After every run, timeout, interruption, crash, or failed command, check again
for leftover training or benchmark processes. A timed-out command must be
treated as suspicious until the process table confirms that no project
`train.exe` remains alive.

Never leave overlapping training/probe runs by accident. Overlap can corrupt
measurements, saturate the GPU, exhaust RAM/VRAM, and make the user's PC
unusable.

## Required Ramp Protocol

For any new training path, kernel path, memory layout, batch shape, sequence
length, or cache mode:

1. start at a known-safe small configuration, usually batch `1` or `4`;
2. record peak VRAM/RAM, step time, loss sanity, and whether the process remains
   responsive;
3. increase only by a bounded step, for example `4 -> 8 -> 16 -> 32`, based on
   the measured headroom;
4. stop before the system approaches unsafe memory pressure;
5. never jump from small unmeasured cases to huge batches;
6. never use a large batch because it "should fit" by rough arithmetic alone.

## Safety Envelope

This is the user's active PC. Treat it as shared interactive hardware, not as an
exclusive benchmark node. Even when the user is not actively typing, do not run
tests that can freeze the display, interrupt video playback, trigger driver
resets, or consume the machine without explicit approval.

Treat these as stop signals:

- OOM, driver reset, system freeze, reboot, or severe UI unresponsiveness;
- unexplained NaN/Inf after a batch/sequence increase;
- VRAM close enough to full that allocator fragmentation can crash later;
- host RAM pressure that risks paging or system instability;
- a new path with unmeasured cache/materialization behavior.

## Interactive-PC Test Limits

For new or recently modified Vulkan/GPU training paths:

1. compile first; do not jump directly into repeated training smokes;
2. run at most one minimal smoke before inspecting logs;
3. use batch `1`, the smallest available step count, and a short timeout until
   the path is known stable;
4. leave a visible safety margin in VRAM/RAM instead of aiming near the limit;
5. do not chain multiple GPU smokes back-to-back when one result already shows
   a crash, abnormal loss, or unclear activation;
6. if the display blanks, video playback stalls, the driver resets, or the UI
   becomes unresponsive, stop operational testing and downgrade to code review,
   build-only checks, or CPU-only diagnostics.

Scaling up is allowed only after the prior run was responsive and produced a
clean activation/loss/memory record. The default posture is "small, inspect,
then decide", not "run a sweep".

## Reporting

When proposing a larger run, report:

- last tested batch/sequence;
- measured peak VRAM/RAM;
- measured step time;
- proposed next batch/sequence;
- expected headroom;
- stop condition.

## Research Meaning

The policy should not suppress hardware efficiency. It should prevent blind
overshoot. The correct behavior is to flex upward safely, not to stay forever at
batch `4`.
