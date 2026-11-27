/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * A candidate's contact sites on neuroglancer's async-computation workers
 * (up to min(12, cores), shared with chunk decoding), so the Twig Selector's
 * contact search runs off the main thread and several candidates at once.
 */

import { asyncComputation } from "#src/async_computation/index.js";

export type ContactSites = Array<[number, [number, number, number]]>;

export interface ContactSearch {
  sites: ContactSites;
  /**
   * The pair has a one-voxel interface somewhere in the crop (the runner's
   * pair_touches). When it does not, its sites came from the widened one.
   */
  touches: boolean;
}

export const twigContactSites = asyncComputation<
  (
    /** The candidate's crop of the label cutout, x fastest. */
    lab: Uint32Array<ArrayBuffer>,
    shape: [number, number, number],
    target: number,
    candidate: number,
    minVoxels: number,
    maxSites: number,
    /** Tried first; if it finds no contact and `fallbackGap` is set, that. */
    gap: number,
    fallbackGap: number | undefined,
    /** The crop's corner in the cutout, added to the centroids. */
    origin: [number, number, number],
  ) => ContactSearch
>("twigContactSites");
