package io.geoalpha.service.domain;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Structured intelligence product returned to the operator after a
 * mission completes.  All fields are nullable so partial products can
 * be shipped if some analytics fail.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record IntelProduct(
        UUID missionId,
        Instant generatedAt,
        Map<String, Object> regime,
        Map<String, Object> changePoint,
        Map<String, Object> tasking,
        Map<String, Object> uncertainty,
        Map<String, Object> anomaly,
        List<String> warnings
) {}
