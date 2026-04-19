package io.geoalpha.service.domain;

import org.springframework.stereotype.Repository;

import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * In-memory mission store.  Swapped for a JPA implementation in
 * production - I just don't want a Postgres dependency in the demo.
 */
@Repository
public class MissionRepository {

    private final ConcurrentMap<UUID, Mission> store = new ConcurrentHashMap<>();
    private final ConcurrentMap<UUID, IntelProduct> products = new ConcurrentHashMap<>();

    public Mission save(Mission m) {
        store.put(m.id(), m);
        return m;
    }

    public Optional<Mission> findById(UUID id) {
        return Optional.ofNullable(store.get(id));
    }

    public List<Mission> findAll() {
        return store.values().stream()
                .sorted(Comparator.comparing(Mission::submittedAt,
                        Comparator.nullsLast(Comparator.naturalOrder())).reversed())
                .toList();
    }

    public List<Mission> findByState(Mission.State state) {
        return store.values().stream()
                .filter(m -> m.state() == state)
                .sorted(Comparator.comparingInt((Mission m) -> -m.priority().ordinal())
                        .thenComparing(Mission::submittedAt,
                                Comparator.nullsLast(Comparator.naturalOrder())))
                .toList();
    }

    public void putProduct(UUID missionId, IntelProduct product) {
        products.put(missionId, product);
    }

    public Optional<IntelProduct> productOf(UUID missionId) {
        return Optional.ofNullable(products.get(missionId));
    }

    public int total() {
        return store.size();
    }
}
