#pragma once
#ifndef KITTY_ENABLE_SYZYGY
#define KITTY_ENABLE_SYZYGY 0
#endif
#if KITTY_ENABLE_SYZYGY != 0 && KITTY_ENABLE_SYZYGY != 1
#error KITTY_ENABLE_SYZYGY must be 0 or 1
#endif
