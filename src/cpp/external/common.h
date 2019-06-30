#pragma once

// std includes
#include <functional>
#include <type_traits>

/*
 * to access c11 enums like int, see:

 * https://stackoverflow.com/questions/8357240/how-to-automatically-convert-strongly-typed-enum-into-int

*/

template <typename E>
constexpr auto to_underlying(E e) noexcept {
    return static_cast <std::underlying_type_t<E>>(e);
}

template <typename E, typename T>
constexpr inline typename std::enable_if <std::is_enum<E>::value &&
                                          std::is_integral<T>::value,
                                          E>::type to_enum(T value) noexcept {
    return static_cast <E>(value);
}


namespace K273 {

    template <typename Type_T>  class ObjectPool {
      private:
        struct FreeList {
            FreeList *next;
            Type_T element;
        };

      public:
        ObjectPool(int block_size=100, std::function <void (Type_T*)> init=nullptr) :
            // DEBUG ONLY
            block_size(block_size),
            head_of_free_list(nullptr),
            init_fn(init) {
         }

        ~ObjectPool() {
            // XXX TODO:  track blocks, and delete them...
            // currently leaks
        }

      public:
        // Get a Type_T element from the free list
        template <class ... ArgTypes> Type_T* acquire(ArgTypes ... args) {
            Type_T* pt_new_element;

            // If head_of_free_list is valid, we'll use and move the head to next
            // element in the free list
            if (this->head_of_free_list != nullptr) {
                pt_new_element = &this->head_of_free_list->element;
                this->head_of_free_list = this->head_of_free_list->next;

            } else {
                // The free list is empty (head_of_free_list is nullptr)

                // Allocate a block of memory (calling constructors)
                FreeList* pt_block = new FreeList[this->block_size];

                // XXX just going to assume new is successful...
                if (this->init_fn != nullptr) {
                    for (int ii=0; ii<this->block_size; ii++) {
                        Type_T& element = pt_block[ii].element;
                        this->init_fn(&element);
                    }
                }

                // Set the new element to the first allocated element
                pt_new_element = &pt_block->element;
                pt_block++;

                // we want to set the rest to head_of_free_list and link them up
                this->head_of_free_list = pt_block;

                // Form a new free list by iterating through and linking the chunks
                // together (except for the first one which we have used and and
                // the last which will be null terminated, therefore block_size - 2).

                for (int ii=0; ii < this->block_size - 3; ii++) {
                    FreeList* cur = pt_block++;
                    cur->next = pt_block;
                }

                // Terminate the last one
                pt_block->next = nullptr;
            }

            pt_new_element->acquire(args ...);
            return pt_new_element;
        }

        // Return a Type_T element to the free list
        void release(Type_T* dead_object) {
            // Make returned object head of free list
            FreeList* carcass = (FreeList*) (((char*) dead_object) - sizeof(FreeList*));
            carcass->next = this->head_of_free_list;
            this->head_of_free_list = carcass;
        }

      private:
        int block_size;
        FreeList* head_of_free_list;
        std::function <void (Type_T*)> init_fn;
    };

    /////////////////////////////////////////////////////////////////////////////

}
