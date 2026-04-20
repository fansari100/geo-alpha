package io.geoalpha.service.client;

import com.fasterxml.jackson.databind.JsonNode;
import io.geoalpha.service.domain.IntelProduct;
import io.geoalpha.service.domain.Mission;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.util.retry.Retry;

import java.time.Duration;
import java.time.Instant;
import java.util.*;

/**
 * Reactive client to the Python analytics gateway.
 *
 * Each Mission.Analytic maps to a single REST call; we fan them out
 * concurrently with Mono.zip so the service-side wallclock is bounded
 * by the slowest analytic, not the sum.
 */
@Service
public class AnalyticsClient {

    private static final Logger LOG = LoggerFactory.getLogger(AnalyticsClient.class);

    private final WebClient client;

    public AnalyticsClient(@Value("${geoalpha.api.base-url:http://localhost:8000}") String baseUrl) {
        LOG.info("Analytics gateway base URL: {}", baseUrl);
        this.client = WebClient.builder().baseUrl(baseUrl).build();
    }

    public Mono<IntelProduct> runFor(Mission mission, double[] sampleSeries, double[] sampleScores) {
        var analytics = mission.analytics();
        List<Mono<Map.Entry<Mission.Analytic, JsonNode>>> calls = new ArrayList<>();
        for (Mission.Analytic a : analytics) {
            switch (a) {
                case REGIME_DETECT  -> calls.add(call(a, "/quant/regime",
                        Map.of("series", boxed(sampleSeries), "n_states", 2)));
                case CHANGE_POINT   -> calls.add(call(a, "/quant/change_point",
                        Map.of("series", boxed(sampleSeries), "hazard_lambda", 200.0)));
                case TASKING        -> calls.add(call(a, "/quant/tasking",
                        Map.of("targets", List.of(
                                Map.of("name", mission.aoi().name(), "value", 10.0,
                                        "dwell_max", 30.0, "priority", mission.priority().name())),
                                "total_budget_s", 60.0)));
                case UNCERTAINTY    -> calls.add(call(a, "/quant/uncertainty",
                        Map.of("toa", List.of(0.18, 0.21, 0.16), "n_samples", 256)));
                case ANOMALY        -> calls.add(call(a, "/quant/anomaly",
                        Map.of("scores", boxed(sampleScores),
                                "threshold_quantile", 0.95, "target_far", 1e-4)));
            }
        }
        return Flux.merge(calls).collectMap(Map.Entry::getKey, Map.Entry::getValue)
                .map(byAnalytic -> assemble(mission, byAnalytic));
    }

    private Mono<Map.Entry<Mission.Analytic, JsonNode>> call(Mission.Analytic a, String path, Map<String, Object> body) {
        return client.post().uri(path)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(JsonNode.class)
                .timeout(Duration.ofSeconds(60))
                .retryWhen(Retry.backoff(2, Duration.ofMillis(250)))
                .map(node -> Map.entry(a, node))
                .doOnError(err -> LOG.warn("analytic {} failed: {}", a, err.toString()));
    }

    private static IntelProduct assemble(Mission m, Map<Mission.Analytic, JsonNode> by) {
        Map<String, Object> regime = mapOf(by.get(Mission.Analytic.REGIME_DETECT));
        Map<String, Object> cp     = mapOf(by.get(Mission.Analytic.CHANGE_POINT));
        Map<String, Object> task   = mapOf(by.get(Mission.Analytic.TASKING));
        Map<String, Object> unc    = mapOf(by.get(Mission.Analytic.UNCERTAINTY));
        Map<String, Object> anom   = mapOf(by.get(Mission.Analytic.ANOMALY));
        List<String> warnings = new ArrayList<>();
        for (Mission.Analytic want : m.analytics()) {
            if (!by.containsKey(want)) warnings.add("analytic " + want + " skipped");
        }
        return new IntelProduct(m.id(), Instant.now(), regime, cp, task, unc, anom, warnings);
    }

    private static Map<String, Object> mapOf(JsonNode n) {
        if (n == null) return null;
        var m = new LinkedHashMap<String, Object>();
        n.fields().forEachRemaining(e -> m.put(e.getKey(), e.getValue()));
        return m;
    }

    private static List<Double> boxed(double[] xs) {
        var out = new ArrayList<Double>(xs.length);
        for (double x : xs) out.add(x);
        return out;
    }
}
