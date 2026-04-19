package io.geoalpha.service.domain;

import com.fasterxml.jackson.annotation.JsonInclude;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

/**
 * A single analyst tasking - what to image, when, and which analytics
 * to run on the result.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record Mission(
        UUID id,
        @NotNull @Size(min = 1, max = 64) String name,
        @NotNull AreaOfInterest aoi,
        @NotNull Priority priority,
        @NotNull List<Analytic> analytics,
        Instant submittedAt,
        Instant scheduledAt,
        Instant completedAt,
        State state,
        String operator
) {
    public enum Priority { ROUTINE, PRIORITY, IMMEDIATE, FLASH }
    public enum State { PENDING, SCHEDULED, RUNNING, COMPLETED, FAILED, CANCELLED }
    public enum Analytic { REGIME_DETECT, CHANGE_POINT, TASKING, UNCERTAINTY, ANOMALY }

    public Mission withState(State s)        { return copy(s, scheduledAt, completedAt); }
    public Mission scheduledNow()            { return copy(state, Instant.now(), completedAt); }
    public Mission completedNow()            { return copy(state, scheduledAt, Instant.now()); }

    private Mission copy(State s, Instant sched, Instant done) {
        return new Mission(id, name, aoi, priority, analytics, submittedAt, sched, done, s, operator);
    }
}
