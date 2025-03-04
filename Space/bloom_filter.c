#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define FNV_PRIME_32 ((uint32_t)0x01000193)
#define FNV_OFFSET_BASIS_32 ((uint32_t)0x811c9dc5)

uint32_t fnv1a_hash_seed(const void *data, size_t len, uint32_t seed) {
    const unsigned char *bytes = (const unsigned char *)data;
    uint32_t hash = seed;
    for (size_t i = 0; i < len; ++i) {
        hash ^= (uint32_t)bytes[i];
        hash *= FNV_PRIME_32;
    }
    return hash;
}

uint32_t fnv1a_hash(const void *data, size_t len) {
    return fnv1a_hash_seed(data, len, FNV_OFFSET_BASIS_32);
}

typedef struct {
    uint8_t *bits;
    size_t num_bits;
    uint32_t num_hashes;
    size_t num_items_added;
    size_t num_bytes;
} BloomFilter;

void set_bit(uint8_t *bits, size_t index) {
    size_t byte_index = index / 8;
    uint8_t bit_mask = (1 << (index % 8));
    bits[byte_index] |= bit_mask;
}

int get_bit(const uint8_t *bits, size_t index) {
    size_t byte_index = index / 8;
    uint8_t bit_mask = (1 << (index % 8));
    return (bits[byte_index] & bit_mask) != 0;
}

BloomFilter* bloom_create(size_t expected_items, double fp_rate) {
    if (expected_items == 0 || fp_rate <= 0.0 || fp_rate >= 1.0) {
        fprintf(stderr, "Error: Invalid parameters for bloom_create.\n");
        return NULL;
    }

    double num_bits_d = -((double)expected_items * log(fp_rate)) / (log(2.0) * log(2.0));
    size_t num_bits = (size_t)ceil(num_bits_d);
    if (num_bits == 0) num_bits = 1;

    double num_hashes_d = ((double)num_bits / (double)expected_items) * log(2.0);
    uint32_t num_hashes = (uint32_t)ceil(num_hashes_d);
     if (num_hashes == 0) num_hashes = 1;

    BloomFilter *filter = (BloomFilter*)malloc(sizeof(BloomFilter));
    if (!filter) {
        perror("Failed to allocate BloomFilter struct");
        return NULL;
    }

    size_t num_bytes = (num_bits + 7) / 8;
    filter->bits = (uint8_t*)calloc(num_bytes, sizeof(uint8_t));
    if (!filter->bits) {
        perror("Failed to allocate bit array");
        free(filter);
        return NULL;
    }

    filter->num_bits = num_bits;
    filter->num_hashes = num_hashes;
    filter->num_items_added = 0;
    filter->num_bytes = num_bytes;

    printf("Bloom Filter Created:\n");
    printf("  Expected Items (n):         %zu\n", expected_items);
    printf("  FP Rate Target (P):         %.4f (1 in %.0f)\n", fp_rate, 1.0/fp_rate);
    printf("  Calculated Bits (m):        %zu\n", filter->num_bits);
    printf("  Calculated Hashes (k):      %u\n", filter->num_hashes);
    printf("  RAM Usage for Bit Array:    %zu bytes", filter->num_bytes);
    if (filter->num_bytes >= 1024 * 1024) {
        printf(" (%.2f MB)\n", (double)filter->num_bytes / (1024.0 * 1024.0));
    } else if (filter->num_bytes >= 1024) {
         printf(" (%.2f KB)\n", (double)filter->num_bytes / 1024.0);
    } else {
        printf("\n");
    }
    if (expected_items > 0) {
        printf("  Bits per Expected Item:     %.2f\n", (double)filter->num_bits / expected_items);
    }
    printf("----------------------------------\n");

    return filter;
}

void bloom_destroy(BloomFilter *filter) {
    if (filter) {
        free(filter->bits);
        free(filter);
    }
}

void bloom_get_hashes(const BloomFilter *filter, const void *item, size_t item_len, uint32_t *hash_indices) {
    uint32_t h1 = fnv1a_hash_seed(item, item_len, FNV_OFFSET_BASIS_32);
    uint32_t h2 = fnv1a_hash_seed(item, item_len, FNV_PRIME_32);

    for (uint32_t i = 0; i < filter->num_hashes; ++i) {
        hash_indices[i] = (h1 + i * h2) % filter->num_bits;
    }
}

void bloom_add(BloomFilter *filter, const void *item, size_t item_len) {
    if (!filter || !item || item_len == 0) return;

    uint32_t *hash_indices = (uint32_t*)malloc(filter->num_hashes * sizeof(uint32_t));
    if (!hash_indices) {
        perror("Failed to allocate hash indices array in add");
        return;
    }

    bloom_get_hashes(filter, item, item_len, hash_indices);

    for (uint32_t i = 0; i < filter->num_hashes; ++i) {
        set_bit(filter->bits, hash_indices[i]);
    }

    filter->num_items_added++;
    free(hash_indices);
}

int bloom_check(const BloomFilter *filter, const void *item, size_t item_len) {
    if (!filter || !item || item_len == 0) return 0;

    uint32_t *hash_indices = (uint32_t*)malloc(filter->num_hashes * sizeof(uint32_t));
     if (!hash_indices) {
        perror("Failed to allocate hash indices array in check");
        return 0;
    }

    bloom_get_hashes(filter, item, item_len, hash_indices);

    for (uint32_t i = 0; i < filter->num_hashes; ++i) {
        if (!get_bit(filter->bits, hash_indices[i])) {
            free(hash_indices);
            return 0;
        }
    }

    free(hash_indices);
    return 1;
}

int main() {
    size_t n_minority_set = 50000;
    double P_allowable_error = 1.0/16.0;

    clock_t start_create = clock();
    BloomFilter *filter = bloom_create(n_minority_set, P_allowable_error);
    clock_t end_create = clock();
    if (!filter) {
        return 1;
    }
    double time_create = ((double)(end_create - start_create)) / CLOCKS_PER_SEC;
    printf("Time to create filter: %.6f seconds\n", time_create);

    printf("\nAdding %zu items to the filter...\n", n_minority_set);
    char word_buffer[50];
    clock_t start_add = clock();
    for (size_t i = 0; i < n_minority_set; ++i) {
        snprintf(word_buffer, sizeof(word_buffer), "minority_word_%zu", i);
        bloom_add(filter, word_buffer, strlen(word_buffer));
    }
    clock_t end_add = clock();
    double time_add_total = ((double)(end_add - start_add)) / CLOCKS_PER_SEC;
    double time_add_per_item = time_add_total / n_minority_set;

    printf("Finished adding items. Total added: %zu\n", filter->num_items_added);
    printf("Time to add all items: %.6f seconds\n", time_add_total);
    printf("Average time per add operation: %.9f seconds (%.3f ns)\n",
           time_add_per_item, time_add_per_item * 1e9);

    printf("\n--- Testing Membership ---\n");

    int found_count = 0;
    size_t test_in_set_count = 10000;
    printf("Testing %zu items known to be IN the set:\n", test_in_set_count);
    clock_t start_check_in = clock();
    for (size_t i = 0; i < test_in_set_count; ++i) {
        snprintf(word_buffer, sizeof(word_buffer), "minority_word_%zu", i % n_minority_set);
        if (bloom_check(filter, word_buffer, strlen(word_buffer))) {
            found_count++;
        } else {
             printf("  ERROR: Item '%s' known to be in set was NOT found!\n", word_buffer);
        }
    }
    clock_t end_check_in = clock();
    double time_check_in_total = ((double)(end_check_in - start_check_in)) / CLOCKS_PER_SEC;
    double time_check_in_per_item = time_check_in_total / test_in_set_count;

    printf("  Found %d out of %zu items tested (should be 100%% found).\n", found_count, test_in_set_count);
    printf("  Time to check items IN set: %.6f seconds\n", time_check_in_total);
    printf("  Average time per check (in set): %.9f seconds (%.3f ns)\n",
           time_check_in_per_item, time_check_in_per_item * 1e9);

    int false_positive_count = 0;
    size_t test_not_in_set_count = 100000;
    printf("\nTesting %zu items known to be NOT in the set:\n", test_not_in_set_count);
    clock_t start_check_out = clock();
    for (size_t i = 0; i < test_not_in_set_count; ++i) {
        snprintf(word_buffer, sizeof(word_buffer), "majority_word_%zu", i);
        if (bloom_check(filter, word_buffer, strlen(word_buffer))) {
            false_positive_count++;
        }
    }
    clock_t end_check_out = clock();
    double time_check_out_total = ((double)(end_check_out - start_check_out)) / CLOCKS_PER_SEC;
    double time_check_out_per_item = time_check_out_total / test_not_in_set_count;

    double observed_fp_rate = (double)false_positive_count / test_not_in_set_count;
    printf("  Found %d false positives out of %zu items tested.\n", false_positive_count, test_not_in_set_count);
    printf("  Observed False Positive Rate:   %.6f\n", observed_fp_rate);
    printf("  Target False Positive Rate:     %.6f\n", P_allowable_error);
    printf("  Time to check items NOT in set: %.6f seconds\n", time_check_out_total);
    printf("  Average time per check (not in set): %.9f seconds (%.3f ns)\n",
           time_check_out_per_item, time_check_out_per_item * 1e9);

    double k_act = (double)filter->num_hashes;
    double m_act = (double)filter->num_bits;
    double n_act = (double)filter->num_items_added;
    double theoretical_fp = 0.0;
    if (m_act > 0 && n_act > 0) {
        theoretical_fp = pow(1.0 - exp(-k_act * n_act / m_act), k_act);
    }
    printf("\n  Theoretical FP Rate (calculated): %.6f\n", theoretical_fp);

    bloom_destroy(filter);
    printf("\nBloom filter destroyed.\n");

    return 0;
}