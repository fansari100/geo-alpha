package io.geoalpha.service.scheduler;

import io.geoalpha.service.client.AnalyticsClient;
import io.geoalpha.service.domain.Mission;
import io.geoalpha.service.domain.MissionRepository;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.util.Random;

/**
 * Polls the mission repository at a fixed cadence, picks the highest-
 * priority pending mission, dispatches it to the analytics gateway,
 * and updates the lifecycle state on completion.
 */
@Service
public class MissionScheduler {

    private static final Logger LOG = LoggerFactory.getLogger(MissionScheduler.class);

    private final MissionRepository repo;
    private final AnalyticsClient analytics;
    private final Counter completed;
    private final Counter failed;
    private final Timer wallclock;

    public MissionScheduler(MissionRepository repo, AnalyticsClient analytics, MeterRegistry registry) {
        this.repo = repo;
        this.analytics = analytics;
        this.completed = Counter.builder("geoalpha.missions.completed").register(registry);
        this.failed = Counter.builder("geoalpha.missions.failed").register(registry);
        this.wallclock = Timer.builder("geoalpha.missions.wallclock").register(registry);
    }

    @Scheduled(fixedDelay = 1500)
    public void tick() {
        var inflight = repo.findByState(Mission.State.RUNNING).size();
        if (inflight >= 8) return;
        var pending = repo.findByState(Mission.State.SCHEDULED);
        if (pending.isEmpty()) return;

        Mission next = pending.get(0);
        Mission running = repo.save(next.withState(Mission.State.RUNNING).scheduledNow());

        // Synthetic series + scores - in production these would come from
        // pre-staged sensor product (TOA radiance -> NDVI series -> RX scores).
        double[] series = synthSeries(running.id().getMostSignificantBits(), 256);
        double[] scores = synthScores(running.id().getLeastSignificantBits(), 4096);

        Timer.Sample sample = Timer.start();
        analytics.runFor(running, series, scores)
                .doOnSuccess(prod -> {
                    repo.putProduct(running.id(), prod);
                    repo.save(running.withState(Mission.State.COMPLETED).completedNow());
                    completed.increment();
                    sample.stop(wallclock);
                    LOG.info("Mission {} done", running.id());
                })
                .doOnError(err -> {
                    repo.save(running.withState(Mission.State.FAILED).completedNow());
                    failed.increment();
                    sample.stop(wallclock);
                    LOG.warn("Mission {} failed: {}", running.id(), err.toString());
                })
                .subscribe();
    }

    private static double[] synthSeries(long seed, int n) {
        Random rng = new Random(seed);
        double[] x = new double[n];
        double mean1 = 0.55, mean2 = 0.30;
        int shift = n / 2;
        for (int i = 0; i < n; i++) {
            double m = i < shift ? mean1 : mean2;
            x[i] = m + 0.15 * Math.sin(i / 12.0) + 0.03 * rng.nextGaussian();
        }
        return x;
    }

    private static double[] synthScores(long seed, int n) {
        Random rng = new Random(seed);
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = Math.abs(rng.nextGaussian());
        }
        return x;
    }
}
